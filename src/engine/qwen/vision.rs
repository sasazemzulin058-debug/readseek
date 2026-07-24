// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (c) 2026 Jarkko Sakkinen

//! Fixed CPU vision encoder for the pinned Qwen3-VL-2B multimodal projector.

/// Narrow a small model dimension/index to `f32` losslessly. Exact for any
/// value that fits `u16` (the 24-bit `f32` mantissa holds every `u16`).
fn dim_to_f32(value: usize) -> f32 {
    f32::from(u16::try_from(value).expect("dimension/index exceeds u16"))
}

/// Narrow a model dimension to `u32` for GGUF metadata comparison.
fn dim_to_u32(value: usize) -> u32 {
    u32::try_from(value).expect("dimension fits u32")
}

/// Lossy `usize -> f32` for image pixel counts and dimensions that may exceed
/// `u16`. Used only in geometric scale-factor math (`sqrt`, `/beta`) where the
/// resulting float is consumed approximately, so precision loss is harmless.
use num_traits::ToPrimitive;

fn count_to_f32(value: usize) -> f32 {
    value.to_f32().unwrap_or(0.0)
}

/// Floor a non-negative `f32` (bounded by the caller) to `usize`. The callers
/// pass clamped/positive values (coordinates and aligned dimensions).
fn floor_to_usize(value: f32) -> usize {
    value.floor().to_usize().unwrap_or(0)
}

/// Ceiling of a non-negative `f32` (bounded by the caller) to `usize`. The
/// callers pass clamped/positive values (coordinates and aligned dimensions).
fn ceil_to_usize(value: f32) -> usize {
    value.ceil().to_usize().unwrap_or(0)
}

use std::path::Path;

use anyhow::{Context as _, Result, ensure};
use image::imageops::FilterType;
use image::{RgbImage, load_from_memory};
use rayon::prelude::*;

use super::gguf::{Gguf, Tensor, TensorType};
use super::kernels::{F32DotKernel, add_bias, gelu, layer_norm, matrix_matrix, vector_add};

const PATCH_SIZE: usize = 16;
const MERGE_SIZE: usize = 2;
const ALIGN_SIZE: usize = PATCH_SIZE * MERGE_SIZE;
const CHANNELS: usize = 3;
const PATCH_VALUES: usize = PATCH_SIZE * PATCH_SIZE * CHANNELS;
const POSITION_SIDE: usize = 48;
const EMBEDDING_WIDTH: usize = 1024;
const FFN_WIDTH: usize = 4096;
const HEAD_COUNT: usize = 16;
const HEAD_WIDTH: usize = EMBEDDING_WIDTH / HEAD_COUNT;
const QKV_WIDTH: usize = EMBEDDING_WIDTH * 3;
const PROJECTED_WIDTH: usize = 2048;
const DEEPSTACK_LAYERS: [usize; 3] = [5, 11, 17];
const OUTPUT_WIDTH: usize = PROJECTED_WIDTH * (1 + DEEPSTACK_LAYERS.len());
const LAYER_COUNT: usize = 24;
const MIN_IMAGE_TOKENS: usize = 8;
const MAX_IMAGE_TOKENS: usize = 4096;
const LAYER_NORM_EPSILON: f32 = 1.0e-6;
const ATTENTION_SCALE: f32 = 0.125;
const ROPE_THETA: f32 = 10_000.0;
const ATTENTION_KEY_TILE: usize = 64;
const CACHE_LINE_SIZE: usize = 64;

/// Input accepted by [`VisionModel::encode_input`].
#[derive(Clone, Copy)]
pub enum VisionInput<'a> {
    /// PNG, JPEG, GIF, WebP, BMP, or TIFF bytes decoded by the `image` crate.
    Encoded(&'a [u8]),
    /// Packed row-major RGB8 pixels.
    Rgb {
        width: u32,
        height: u32,
        pixels: &'a [u8],
    },
}

/// Decoder-facing spatial reduction applied after Qwen's trained patch merger.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpatialReduction {
    MergeHalf,
    PruneQuarter,
    None,
}

/// Spatially merged image tokens and their original grid locations.
#[derive(Debug)]
pub struct VisionEmbedding {
    /// Row-major values with shape `token_count x 8192`.
    pub values: Vec<f32>,
    pub positions: Vec<[usize; 2]>,
    pub masses: Vec<u32>,
    pub grid_width: usize,
    pub grid_height: usize,
    pub token_count: usize,
    pub original_token_count: usize,
}

/// Loaded Qwen3-VL-2B `Q8_0` multimodal projector.
pub struct VisionModel {
    gguf: Gguf,
}

#[derive(Clone, Copy)]
struct TokenCluster {
    anchor: usize,
    partner: Option<usize>,
}

/// Cache-line-aligned planar RGB bitmap with a power-of-two row stride.
///
/// `width` and `height` are the logical resized dimensions used by the model.
/// Width padding is storage-only, so it does not add vision tokens; height is
/// never padded.
struct Bitmap {
    storage: Vec<u8>,
    pixel_offset: usize,
    pixel_len: usize,
    width: usize,
    height: usize,
    width_shift: u32,
    plane_len: usize,
}

impl Bitmap {
    fn from_image(image: RgbImage) -> Result<Self> {
        let width = image.width() as usize;
        let height = image.height() as usize;
        let padded_width = width
            .checked_next_power_of_two()
            .context("padded bitmap width overflow")?;
        let plane_len = padded_width
            .checked_mul(height)
            .context("padded bitmap plane size overflow")?;
        let pixel_len = plane_len
            .checked_mul(CHANNELS)
            .context("padded bitmap size overflow")?;
        let storage_len = pixel_len
            .checked_add(CACHE_LINE_SIZE - 1)
            .context("aligned bitmap allocation size overflow")?;
        let mut storage = vec![0; storage_len];
        let pixel_offset = storage.as_ptr().align_offset(CACHE_LINE_SIZE);
        ensure!(
            pixel_offset < CACHE_LINE_SIZE,
            "could not align bitmap storage to {CACHE_LINE_SIZE} bytes"
        );

        let source_stride = width
            .checked_mul(CHANNELS)
            .context("bitmap source stride overflow")?;
        let source = image.into_raw();
        let pixels = &mut storage[pixel_offset..pixel_offset + pixel_len];
        let width_shift = padded_width.trailing_zeros();
        for (y, source) in source.chunks_exact(source_stride).enumerate() {
            let destination_row = y << width_shift;
            for (x, source) in source.chunks_exact(CHANNELS).enumerate() {
                let destination = destination_row + x;
                pixels[destination] = source[0];
                pixels[plane_len + destination] = source[1];
                pixels[plane_len * 2 + destination] = source[2];
            }
        }

        Ok(Self {
            storage,
            pixel_offset,
            pixel_len,
            width,
            height,
            width_shift,
            plane_len,
        })
    }

    fn pixels(&self) -> &[u8] {
        &self.storage[self.pixel_offset..self.pixel_offset + self.pixel_len]
    }
}

impl VisionModel {
    /// Load and fully validate the fixed Qwen3-VL-2B mmproj GGUF.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let gguf = Gguf::load(path).context("load Qwen3-VL vision projector")?;
        validate_metadata(&gguf)?;
        validate_tensors(&gguf)?;
        Ok(Self { gguf })
    }

    /// Encode either supported input representation.
    pub fn encode_input(
        &self,
        input: VisionInput<'_>,
        image_max_tokens: usize,
        reduction: SpatialReduction,
    ) -> Result<VisionEmbedding> {
        ensure!(
            (MIN_IMAGE_TOKENS..=MAX_IMAGE_TOKENS).contains(&image_max_tokens),
            "image_max_tokens must be in {MIN_IMAGE_TOKENS}..={MAX_IMAGE_TOKENS}"
        );
        let image = decode_input(input)?;
        let resized = smart_resize(image, image_max_tokens)?;
        let bitmap = Bitmap::from_image(resized)?;
        let patch_width = bitmap.width / PATCH_SIZE;
        let patch_height = bitmap.height / PATCH_SIZE;
        let token_count = patch_width
            .checked_mul(patch_height)
            .context("vision patch count overflow")?;

        let mut hidden = self.patch_embeddings(&bitmap, patch_width, patch_height)?;
        drop(bitmap);
        let positions = self.position_embeddings(patch_width, patch_height)?;
        vector_add(&mut hidden, &positions)?;

        let mut deepstack = Vec::with_capacity(DEEPSTACK_LAYERS.len());
        let rope = VisionRope::new(token_count, patch_width)?;
        for layer in 0..LAYER_COUNT {
            hidden = self.transformer_layer(hidden, token_count, layer, &rope)?;
            if DEEPSTACK_LAYERS.contains(&layer) {
                deepstack.push(self.project_deepstack(&hidden, token_count, layer)?);
            }
        }

        let post_weight = f32_values(&self.gguf, "v.post_ln.weight")?;
        let post_bias = f32_values(&self.gguf, "v.post_ln.bias")?;
        let hidden = layer_norm(
            &hidden,
            EMBEDDING_WIDTH,
            post_weight,
            post_bias,
            LAYER_NORM_EPSILON,
        )?;
        validate_grouped_rows(&hidden, token_count, EMBEDDING_WIDTH)?;
        let merged_tokens = token_count / 4;
        let mut main = linear(
            &self.gguf,
            "mm.0.weight",
            "mm.0.bias",
            &hidden,
            merged_tokens,
        )?;
        gelu(&mut main);
        main = linear(&self.gguf, "mm.2.weight", "mm.2.bias", &main, merged_tokens)?;

        let mut values = vec![0.0; merged_tokens * OUTPUT_WIDTH];
        values
            .par_chunks_mut(OUTPUT_WIDTH)
            .enumerate()
            .for_each(|(token, row)| {
                row[..PROJECTED_WIDTH]
                    .copy_from_slice(&main[token * PROJECTED_WIDTH..(token + 1) * PROJECTED_WIDTH]);
                for (stream, features) in deepstack.iter().enumerate() {
                    let output_start = (stream + 1) * PROJECTED_WIDTH;
                    let input_start = token * PROJECTED_WIDTH;
                    row[output_start..output_start + PROJECTED_WIDTH]
                        .copy_from_slice(&features[input_start..input_start + PROJECTED_WIDTH]);
                }
            });

        let grid_width = patch_width / MERGE_SIZE;
        let grid_height = patch_height / MERGE_SIZE;
        ensure!(
            merged_tokens == grid_width * grid_height,
            "vision output grid does not match token count"
        );
        Self::reduce_spatial_tokens(&values, grid_width, grid_height, reduction)
    }

    fn reduce_spatial_tokens(
        values: &[f32],
        grid_width: usize,
        grid_height: usize,
        reduction: SpatialReduction,
    ) -> Result<VisionEmbedding> {
        let original_token_count = grid_width
            .checked_mul(grid_height)
            .context("vision output grid size overflow")?;
        ensure!(
            values.len() == original_token_count * OUTPUT_WIDTH,
            "vision output values do not match grid"
        );

        let mut clusters = Vec::with_capacity(original_token_count);
        match reduction {
            SpatialReduction::None => {
                clusters.extend((0..original_token_count).map(|anchor| TokenCluster {
                    anchor,
                    partner: None,
                }));
            }
            SpatialReduction::PruneQuarter => {
                for tile_y in (0..grid_height).step_by(2) {
                    for tile_x in (0..grid_width).step_by(2) {
                        let tile = Self::tile_indices(tile_x, tile_y, grid_width, grid_height);
                        let dropped =
                            (tile.len() == 4).then(|| Self::closest_pair(values, &tile).1);
                        clusters.extend(
                            tile.into_iter()
                                .filter(|index| Some(*index) != dropped)
                                .map(|anchor| TokenCluster {
                                    anchor,
                                    partner: None,
                                }),
                        );
                    }
                }
            }
            SpatialReduction::MergeHalf => {
                for row in 0..grid_height {
                    for column in (0..grid_width).step_by(2) {
                        let anchor = row * grid_width + column;
                        let partner = (column + 1 < grid_width).then_some(anchor + 1);
                        clusters.push(TokenCluster { anchor, partner });
                    }
                }
            }
        }
        clusters.sort_unstable_by_key(|cluster| cluster.anchor);

        let token_count = clusters.len();
        let mut reduced = Vec::with_capacity(token_count * OUTPUT_WIDTH);
        let mut positions = Vec::with_capacity(token_count);
        let mut masses = Vec::with_capacity(token_count);
        for cluster in clusters {
            let start = cluster.anchor * OUTPUT_WIDTH;
            let anchor = &values[start..start + OUTPUT_WIDTH];
            if let Some(partner) = cluster.partner {
                let start = partner * OUTPUT_WIDTH;
                let partner = &values[start..start + OUTPUT_WIDTH];
                reduced.extend(
                    anchor
                        .iter()
                        .zip(partner)
                        .map(|(left, right)| (left + right) * 0.5),
                );
                masses.push(2);
            } else {
                reduced.extend_from_slice(anchor);
                masses.push(1);
            }
            positions.push([cluster.anchor / grid_width, cluster.anchor % grid_width]);
        }

        Ok(VisionEmbedding {
            values: reduced,
            positions,
            masses,
            grid_width,
            grid_height,
            token_count,
            original_token_count,
        })
    }

    fn tile_indices(
        tile_x: usize,
        tile_y: usize,
        grid_width: usize,
        grid_height: usize,
    ) -> Vec<usize> {
        let mut indices = Vec::with_capacity(4);
        for y in tile_y..(tile_y + 2).min(grid_height) {
            for x in tile_x..(tile_x + 2).min(grid_width) {
                indices.push(y * grid_width + x);
            }
        }
        indices
    }

    fn closest_pair(values: &[f32], indices: &[usize]) -> (usize, usize) {
        let mut best = (indices[0], indices[1]);
        let mut best_score = Self::token_similarity(values, best.0, best.1);
        for left in 0..indices.len() {
            for right in left + 1..indices.len() {
                let pair = (indices[left], indices[right]);
                let score = Self::token_similarity(values, pair.0, pair.1);
                if score > best_score {
                    best = pair;
                    best_score = score;
                }
            }
        }
        best
    }

    fn token_similarity(values: &[f32], left: usize, right: usize) -> f32 {
        let left_start = left * OUTPUT_WIDTH;
        let right_start = right * OUTPUT_WIDTH;
        let left = &values[left_start..left_start + PROJECTED_WIDTH];
        let right = &values[right_start..right_start + PROJECTED_WIDTH];
        let (mut dot, mut left_norm, mut right_norm) = (0.0, 0.0, 0.0);
        for (left, right) in left.iter().zip(right) {
            dot += left * right;
            left_norm += left * left;
            right_norm += right * right;
        }
        let norm = (left_norm * right_norm).sqrt();
        if norm == 0.0 { -1.0 } else { dot / norm }
    }

    fn patch_embeddings(
        &self,
        bitmap: &Bitmap,
        patch_width: usize,
        patch_height: usize,
    ) -> Result<Vec<f32>> {
        let first = f32_values(&self.gguf, "v.patch_embd.weight")?;
        let second = f32_values(&self.gguf, "v.patch_embd.weight.1")?;
        let bias = f32_values(&self.gguf, "v.patch_embd.bias")?;
        let token_count = patch_width * patch_height;
        let pixels = bitmap.pixels();
        let mut output = vec![0.0; token_count * EMBEDDING_WIDTH];

        output
            .par_chunks_mut(EMBEDDING_WIDTH)
            .enumerate()
            .for_each_init(
                || vec![0.0; PATCH_VALUES],
                |patch, (token, row)| {
                    let (patch_x, patch_y) = grouped_patch_coordinates(token, patch_width);
                    let patch_x = patch_x * PATCH_SIZE;
                    let patch_y = patch_y * PATCH_SIZE;
                    for channel in 0..CHANNELS {
                        let channel_start = channel * bitmap.plane_len;
                        for y in 0..PATCH_SIZE {
                            let source_start =
                                channel_start + ((patch_y + y) << bitmap.width_shift) + patch_x;
                            let destination_start = PATCH_SIZE * (y + PATCH_SIZE * channel);
                            let source = &pixels[source_start..source_start + PATCH_SIZE];
                            let destination =
                                &mut patch[destination_start..destination_start + PATCH_SIZE];
                            for (source, destination) in source.iter().zip(destination) {
                                *destination = f32::from(*source) / 127.5 - 1.0;
                            }
                        }
                    }
                    for (channel, value) in row.iter_mut().enumerate() {
                        let start = channel * PATCH_VALUES;
                        let first_row = &first[start..start + PATCH_VALUES];
                        let second_row = &second[start..start + PATCH_VALUES];
                        let first_sum = patch
                            .iter()
                            .zip(first_row)
                            .map(|(input, weight)| input * weight)
                            .sum::<f32>();
                        let second_sum = patch
                            .iter()
                            .zip(second_row)
                            .map(|(input, weight)| input * weight)
                            .sum::<f32>();
                        *value = first_sum + second_sum + bias[channel];
                    }
                },
            );
        Ok(output)
    }

    fn position_embeddings(&self, patch_width: usize, patch_height: usize) -> Result<Vec<f32>> {
        let source = f32_values(&self.gguf, "v.position_embd.weight")?;
        let token_count = patch_width * patch_height;
        let mut output = vec![0.0; token_count * EMBEDDING_WIDTH];
        output
            .par_chunks_mut(EMBEDDING_WIDTH)
            .enumerate()
            .for_each(|(token, row)| {
                let (x, y) = grouped_patch_coordinates(token, patch_width);
                let x_coordinate = interpolation_coordinate(x, patch_width, POSITION_SIDE);
                let y_coordinate = interpolation_coordinate(y, patch_height, POSITION_SIDE);
                let x0 = floor_to_usize(x_coordinate);
                let y0 = floor_to_usize(y_coordinate);
                let x1 = (x0 + 1).min(POSITION_SIDE - 1);
                let y1 = (y0 + 1).min(POSITION_SIDE - 1);
                let x_weight = x_coordinate - dim_to_f32(x0);
                let y_weight = y_coordinate - dim_to_f32(y0);
                for (channel, value) in row.iter_mut().enumerate() {
                    let top_left = position_value(source, x0, y0, channel);
                    let top_right = position_value(source, x1, y0, channel);
                    let bottom_left = position_value(source, x0, y1, channel);
                    let bottom_right = position_value(source, x1, y1, channel);
                    let top = top_left + (top_right - top_left) * x_weight;
                    let bottom = bottom_left + (bottom_right - bottom_left) * x_weight;
                    *value = top + (bottom - top) * y_weight;
                }
            });
        Ok(output)
    }

    fn transformer_layer(
        &self,
        mut hidden: Vec<f32>,
        token_count: usize,
        layer: usize,
        rope: &VisionRope,
    ) -> Result<Vec<f32>> {
        let prefix = format!("v.blk.{layer}");
        let ln1_weight = f32_values(&self.gguf, &format!("{prefix}.ln1.weight"))?;
        let ln1_bias = f32_values(&self.gguf, &format!("{prefix}.ln1.bias"))?;
        let normalized = layer_norm(
            &hidden,
            EMBEDDING_WIDTH,
            ln1_weight,
            ln1_bias,
            LAYER_NORM_EPSILON,
        )?;
        let mut qkv = linear(
            &self.gguf,
            &format!("{prefix}.attn_qkv.weight"),
            &format!("{prefix}.attn_qkv.bias"),
            &normalized,
            token_count,
        )?;
        apply_vision_mrope(&mut qkv, rope)?;
        let attended = attention(&qkv, token_count)?;
        let projected = linear(
            &self.gguf,
            &format!("{prefix}.attn_out.weight"),
            &format!("{prefix}.attn_out.bias"),
            &attended,
            token_count,
        )?;
        vector_add(&mut hidden, &projected)?;

        let ln2_weight = f32_values(&self.gguf, &format!("{prefix}.ln2.weight"))?;
        let ln2_bias = f32_values(&self.gguf, &format!("{prefix}.ln2.bias"))?;
        let normalized = layer_norm(
            &hidden,
            EMBEDDING_WIDTH,
            ln2_weight,
            ln2_bias,
            LAYER_NORM_EPSILON,
        )?;
        let mut feed_forward = linear(
            &self.gguf,
            &format!("{prefix}.ffn_up.weight"),
            &format!("{prefix}.ffn_up.bias"),
            &normalized,
            token_count,
        )?;
        gelu(&mut feed_forward);
        feed_forward = linear(
            &self.gguf,
            &format!("{prefix}.ffn_down.weight"),
            &format!("{prefix}.ffn_down.bias"),
            &feed_forward,
            token_count,
        )?;
        vector_add(&mut hidden, &feed_forward)?;
        Ok(hidden)
    }

    fn project_deepstack(
        &self,
        hidden: &[f32],
        token_count: usize,
        layer: usize,
    ) -> Result<Vec<f32>> {
        validate_grouped_rows(hidden, token_count, EMBEDDING_WIDTH)?;
        let prefix = format!("v.deepstack.{layer}");
        let norm_weight = f32_values(&self.gguf, &format!("{prefix}.norm.weight"))?;
        let norm_bias = f32_values(&self.gguf, &format!("{prefix}.norm.bias"))?;
        let normalized = layer_norm(
            hidden,
            EMBEDDING_WIDTH * 4,
            norm_weight,
            norm_bias,
            LAYER_NORM_EPSILON,
        )?;
        let mut projected = linear(
            &self.gguf,
            &format!("{prefix}.fc1.weight"),
            &format!("{prefix}.fc1.bias"),
            &normalized,
            token_count / 4,
        )?;
        gelu(&mut projected);
        linear(
            &self.gguf,
            &format!("{prefix}.fc2.weight"),
            &format!("{prefix}.fc2.bias"),
            &projected,
            token_count / 4,
        )
    }
}

fn decode_input(input: VisionInput<'_>) -> Result<RgbImage> {
    match input {
        VisionInput::Encoded(bytes) => {
            ensure!(!bytes.is_empty(), "encoded image is empty");
            let image = load_from_memory(bytes).context("decode image")?;
            Ok(image.to_rgb8())
        }
        VisionInput::Rgb {
            width,
            height,
            pixels,
        } => {
            ensure!(
                width != 0 && height != 0,
                "RGB image dimensions must be nonzero"
            );
            let expected = (width as usize)
                .checked_mul(height as usize)
                .and_then(|value| value.checked_mul(CHANNELS))
                .context("RGB image size overflow")?;
            ensure!(
                pixels.len() == expected,
                "RGB image has {} bytes, expected {expected} for {width}x{height}",
                pixels.len()
            );
            RgbImage::from_raw(width, height, pixels.to_vec()).context("construct packed RGB image")
        }
    }
}

fn smart_resize(image: RgbImage, image_max_tokens: usize) -> Result<RgbImage> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let min_pixels = MIN_IMAGE_TOKENS * ALIGN_SIZE * ALIGN_SIZE;
    let max_pixels = image_max_tokens
        .checked_mul(ALIGN_SIZE * ALIGN_SIZE)
        .context("maximum image pixel count overflow")?;
    let mut target_width = round_to_multiple(width, ALIGN_SIZE).max(ALIGN_SIZE);
    let mut target_height = round_to_multiple(height, ALIGN_SIZE).max(ALIGN_SIZE);
    let aligned_pixels = target_width
        .checked_mul(target_height)
        .context("aligned image pixel count overflow")?;

    if aligned_pixels > max_pixels {
        let beta = ((count_to_f32(height) * count_to_f32(width)) / count_to_f32(max_pixels)).sqrt();
        target_height = floor_to_multiple(count_to_f32(height) / beta, ALIGN_SIZE).max(ALIGN_SIZE);
        target_width = floor_to_multiple(count_to_f32(width) / beta, ALIGN_SIZE).max(ALIGN_SIZE);
    } else if aligned_pixels < min_pixels {
        let beta = (count_to_f32(min_pixels) / (count_to_f32(height) * count_to_f32(width))).sqrt();
        target_height = ceil_to_multiple(count_to_f32(height) * beta, ALIGN_SIZE);
        target_width = ceil_to_multiple(count_to_f32(width) * beta, ALIGN_SIZE);
    }
    let (target_width, target_height) =
        adjust_width_to_power_of_two(target_width, target_height, min_pixels)?;
    ensure!(
        target_width.is_multiple_of(ALIGN_SIZE) && target_height.is_multiple_of(ALIGN_SIZE),
        "smart resize dimensions are not aligned"
    );
    let target_width = u32::try_from(target_width).context("resized image width exceeds u32")?;
    let target_height = u32::try_from(target_height).context("resized image height exceeds u32")?;
    if image.width() == target_width && image.height() == target_height {
        return Ok(image);
    }
    Ok(image::imageops::resize(
        &image,
        target_width,
        target_height,
        FilterType::Triangle,
    ))
}

/// Pad a nearer/equidistant next power in [`Bitmap`], but scale the image when
/// the previous width power is strictly nearer. Height follows that scale to
/// preserve the aspect ratio and remains aligned only to the patch grid.
fn adjust_width_to_power_of_two(
    width: usize,
    height: usize,
    min_pixels: usize,
) -> Result<(usize, usize)> {
    if width.is_power_of_two() {
        return Ok((width, height));
    }

    let next_width = width
        .checked_next_power_of_two()
        .context("bitmap width power-of-two overflow")?;
    let previous_width = next_width / 2;
    if next_width - width <= width - previous_width {
        return Ok((width, height));
    }

    let scaled_height_numerator = height
        .checked_mul(previous_width)
        .context("scaled bitmap height overflow")?;
    let aligned_height_denominator = width
        .checked_mul(ALIGN_SIZE)
        .context("aligned bitmap height overflow")?;
    let aligned_height_units = scaled_height_numerator
        .checked_add(aligned_height_denominator / 2)
        .context("rounded bitmap height overflow")?
        / aligned_height_denominator;
    let scaled_height = aligned_height_units
        .max(1)
        .checked_mul(ALIGN_SIZE)
        .context("scaled bitmap height overflow")?;
    let minimum_height = min_pixels
        .div_ceil(previous_width)
        .div_ceil(ALIGN_SIZE)
        .checked_mul(ALIGN_SIZE)
        .context("minimum bitmap height overflow")?;

    Ok((previous_width, scaled_height.max(minimum_height)))
}

fn round_to_multiple(value: usize, factor: usize) -> usize {
    let rounded_up = usize::from(value % factor >= factor.div_ceil(2));
    (value / factor + rounded_up) * factor
}

fn floor_to_multiple(value: f32, factor: usize) -> usize {
    floor_to_usize(value / dim_to_f32(factor)) * factor
}

fn ceil_to_multiple(value: f32, factor: usize) -> usize {
    ceil_to_usize(value / dim_to_f32(factor)) * factor
}

fn grouped_patch_coordinates(token: usize, patch_width: usize) -> (usize, usize) {
    let groups_per_row = patch_width / MERGE_SIZE;
    let group = token / 4;
    let offset = token % 4;
    let group_x = group % groups_per_row;
    let group_y = group / groups_per_row;
    (
        group_x * MERGE_SIZE + offset % MERGE_SIZE,
        group_y * MERGE_SIZE + offset / MERGE_SIZE,
    )
}

fn interpolation_coordinate(index: usize, output_size: usize, input_size: usize) -> f32 {
    let coordinate =
        (dim_to_f32(index) + 0.5) * dim_to_f32(input_size) / dim_to_f32(output_size) - 0.5;
    coordinate.clamp(0.0, dim_to_f32(input_size - 1))
}

fn position_value(values: &[f32], x: usize, y: usize, channel: usize) -> f32 {
    values[(y * POSITION_SIDE + x) * EMBEDDING_WIDTH + channel]
}

struct VisionRope {
    cosine: Vec<f32>,
    sine: Vec<f32>,
    token_count: usize,
}

impl VisionRope {
    fn new(token_count: usize, patch_width: usize) -> Result<Self> {
        let table_len = token_count
            .checked_mul(HEAD_WIDTH / 2)
            .context("vision RoPE table size overflow")?;
        let mut frequencies = [0.0; HEAD_WIDTH / 4];
        for (pair, frequency) in frequencies.iter_mut().enumerate() {
            *frequency = ROPE_THETA.powf(-(dim_to_f32(pair)) / dim_to_f32(HEAD_WIDTH / 4));
        }
        let mut cosine = vec![0.0; table_len];
        let mut sine = vec![0.0; table_len];
        cosine
            .par_chunks_mut(HEAD_WIDTH / 2)
            .zip(sine.par_chunks_mut(HEAD_WIDTH / 2))
            .enumerate()
            .for_each(|(token, (cosine, sine))| {
                let (x, y) = grouped_patch_coordinates(token, patch_width);
                for pair in 0..HEAD_WIDTH / 2 {
                    let position = if pair < HEAD_WIDTH / 4 { y } else { x };
                    let frequency = frequencies[pair % (HEAD_WIDTH / 4)];
                    let angle = dim_to_f32(position) * frequency;
                    cosine[pair] = angle.cos();
                    sine[pair] = angle.sin();
                }
            });
        Ok(Self {
            cosine,
            sine,
            token_count,
        })
    }
}

fn apply_vision_mrope(qkv: &mut [f32], rope: &VisionRope) -> Result<()> {
    ensure!(
        qkv.len().is_multiple_of(QKV_WIDTH),
        "QKV values do not contain complete tokens"
    );
    ensure!(
        qkv.len() / QKV_WIDTH == rope.token_count,
        "QKV token count differs from the vision RoPE table"
    );
    qkv.par_chunks_mut(QKV_WIDTH)
        .zip(
            rope.cosine
                .par_chunks(HEAD_WIDTH / 2)
                .zip(rope.sine.par_chunks(HEAD_WIDTH / 2)),
        )
        .for_each(|(row, (cosine, sine))| {
            for head in 0..HEAD_COUNT {
                rotate_half(
                    &mut row[head * HEAD_WIDTH..(head + 1) * HEAD_WIDTH],
                    cosine,
                    sine,
                );
                let key_start = EMBEDDING_WIDTH + head * HEAD_WIDTH;
                rotate_half(&mut row[key_start..key_start + HEAD_WIDTH], cosine, sine);
            }
        });
    Ok(())
}

fn rotate_half(values: &mut [f32], cosine: &[f32], sine: &[f32]) {
    let (first, second) = values.split_at_mut(HEAD_WIDTH / 2);
    for pair in 0..HEAD_WIDTH / 2 {
        let left = first[pair];
        let right = second[pair];
        first[pair] = left * cosine[pair] - right * sine[pair];
        second[pair] = left * sine[pair] + right * cosine[pair];
    }
}

fn attention(qkv: &[f32], token_count: usize) -> Result<Vec<f32>> {
    attention_with_kernel(qkv, token_count, F32DotKernel::detect())
}

fn attention_with_kernel(
    qkv: &[f32],
    token_count: usize,
    kernel: F32DotKernel,
) -> Result<Vec<f32>> {
    ensure!(token_count != 0, "vision attention has no tokens");
    ensure!(
        qkv.len() == token_count * QKV_WIDTH,
        "vision attention input size differs"
    );
    let output_len = token_count
        .checked_mul(EMBEDDING_WIDTH)
        .context("vision attention output size overflow")?;
    let head_len = token_count
        .checked_mul(HEAD_WIDTH)
        .context("vision attention head size overflow")?;
    let mut keys = vec![0.0; output_len];
    let mut values = vec![0.0; output_len];
    keys.par_chunks_mut(head_len)
        .zip(values.par_chunks_mut(head_len))
        .enumerate()
        .for_each(|(head, (key_head, value_head))| {
            for (token, (key, value)) in key_head
                .chunks_exact_mut(HEAD_WIDTH)
                .zip(value_head.chunks_exact_mut(HEAD_WIDTH))
                .enumerate()
            {
                let token_start = token * QKV_WIDTH;
                let key_start = token_start + EMBEDDING_WIDTH + head * HEAD_WIDTH;
                let value_start = token_start + EMBEDDING_WIDTH * 2 + head * HEAD_WIDTH;
                key.copy_from_slice(&qkv[key_start..key_start + HEAD_WIDTH]);
                value.copy_from_slice(&qkv[value_start..value_start + HEAD_WIDTH]);
            }
        });
    let mut output = vec![0.0; output_len];
    output
        .par_chunks_mut(EMBEDDING_WIDTH)
        .enumerate()
        .for_each_init(
            || [0.0_f32; ATTENTION_KEY_TILE],
            |scores, (query_token, output_token)| {
                for (head, output_head) in output_token.chunks_exact_mut(HEAD_WIDTH).enumerate() {
                    let query_start = query_token * QKV_WIDTH + head * HEAD_WIDTH;
                    let query = &qkv[query_start..query_start + HEAD_WIDTH];
                    let head_start = head * head_len;
                    let key_head = &keys[head_start..head_start + head_len];
                    let value_head = &values[head_start..head_start + head_len];
                    let mut running_max = f32::NEG_INFINITY;
                    let mut running_sum = 0.0_f32;

                    for key_start_token in (0..token_count).step_by(ATTENTION_KEY_TILE) {
                        let tile_len = (token_count - key_start_token).min(ATTENTION_KEY_TILE);
                        let scores = &mut scores[..tile_len];
                        let tile_start = key_start_token * HEAD_WIDTH;
                        let tile_end = tile_start + tile_len * HEAD_WIDTH;
                        let key_tile = &key_head[tile_start..tile_end];
                        for (score, key) in scores.iter_mut().zip(key_tile.chunks_exact(HEAD_WIDTH))
                        {
                            *score = kernel.dot(query, key) * ATTENTION_SCALE;
                        }

                        let tile_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let new_max = running_max.max(tile_max);
                        if running_sum != 0.0 {
                            let previous_scale = (running_max - new_max).exp();
                            running_sum *= previous_scale;
                            for value in output_head.iter_mut() {
                                *value *= previous_scale;
                            }
                        }
                        let value_tile = &value_head[tile_start..tile_end];
                        for (score, value) in scores
                            .iter()
                            .copied()
                            .zip(value_tile.chunks_exact(HEAD_WIDTH))
                        {
                            let weight = (score - new_max).exp();
                            running_sum += weight;
                            kernel.accumulate(output_head, weight, value);
                        }
                        running_max = new_max;
                    }
                    for value in output_head.iter_mut() {
                        *value /= running_sum;
                    }
                }
            },
        );
    ensure!(
        output.iter().all(|value| value.is_finite()),
        "vision attention produced a non-finite value"
    );
    Ok(output)
}

/// Validate the group-major layout produced by `grouped_patch_coordinates`.
/// Consecutive groups of four rows form one merged 2x2 spatial token.
fn validate_grouped_rows(values: &[f32], row_count: usize, width: usize) -> Result<()> {
    ensure!(
        row_count.is_multiple_of(4),
        "row count is not divisible by four"
    );
    let expected = row_count
        .checked_mul(width)
        .context("grouped input size overflow")?;
    ensure!(
        values.len() == expected,
        "grouped input has {} values, expected {expected}",
        values.len()
    );
    Ok(())
}

fn linear(
    gguf: &Gguf,
    weight_name: &str,
    bias_name: &str,
    input: &[f32],
    row_count: usize,
) -> Result<Vec<f32>> {
    let weight = gguf.tensor(weight_name)?;
    let bias = f32_values(gguf, bias_name)?;
    let mut output = matrix_matrix(&weight, input, row_count)
        .with_context(|| format!("apply tensor `{weight_name}`"))?;
    add_bias(&mut output, bias).with_context(|| format!("apply tensor `{bias_name}`"))?;
    Ok(output)
}

fn f32_values<'a>(gguf: &'a Gguf, name: &str) -> Result<&'a [f32]> {
    let tensor = gguf.tensor(name)?;
    tensor
        .f32_slice()
        .with_context(|| format!("read tensor `{name}`"))
}

fn validate_metadata(gguf: &Gguf) -> Result<()> {
    ensure!(
        gguf.architecture() == "clip",
        "GGUF architecture must be `clip`"
    );
    ensure!(
        gguf.string("general.type")? == "mmproj",
        "GGUF type must be `mmproj`"
    );
    ensure!(
        gguf.string("general.basename")? == "qwen3vl",
        "GGUF basename must be `qwen3vl`"
    );
    ensure!(
        gguf.string("general.size_label")? == "407M",
        "GGUF size label must be `407M`"
    );
    ensure!(
        gguf.u32("general.file_type")? == 7,
        "GGUF file type must be Q8_0"
    );
    ensure!(
        gguf.u32("general.quantization_version")? == 2,
        "GGUF quantization version must be 2"
    );
    ensure!(
        gguf.bool("clip.has_vision_encoder")?,
        "GGUF has no vision encoder"
    );
    ensure!(
        gguf.u32("clip.vision.projection_dim")? == dim_to_u32(PROJECTED_WIDTH),
        "invalid projection width"
    );
    ensure!(
        gguf.u32("clip.vision.image_size")? == 768,
        "invalid base image size"
    );
    ensure!(
        gguf.u32("clip.vision.patch_size")? == dim_to_u32(PATCH_SIZE),
        "invalid patch size"
    );
    ensure!(
        gguf.u32("clip.vision.embedding_length")? == dim_to_u32(EMBEDDING_WIDTH),
        "invalid embedding width"
    );
    ensure!(
        gguf.u32("clip.vision.feed_forward_length")? == dim_to_u32(FFN_WIDTH),
        "invalid feed-forward width"
    );
    ensure!(
        gguf.u32("clip.vision.block_count")? == dim_to_u32(LAYER_COUNT),
        "invalid layer count"
    );
    ensure!(
        gguf.u32("clip.vision.attention.head_count")? == dim_to_u32(HEAD_COUNT),
        "invalid attention head count"
    );
    ensure!(
        gguf.string("clip.projector_type")? == "qwen3vl_merger",
        "invalid projector type"
    );
    ensure!(
        gguf.bool("clip.use_gelu")?,
        "vision projector must use GELU"
    );
    ensure!(
        gguf.u32("clip.vision.spatial_merge_size")? == dim_to_u32(MERGE_SIZE),
        "invalid spatial merge size"
    );
    let epsilon = gguf.f32("clip.vision.attention.layer_norm_epsilon")?;
    ensure!(
        epsilon.to_bits() == LAYER_NORM_EPSILON.to_bits(),
        "invalid layer-normalization epsilon {epsilon}"
    );
    let tags = gguf.string_array("general.tags")?;
    ensure!(tags == ["image-text-to-text"], "invalid GGUF tags {tags:?}");
    Ok(())
}

fn validate_tensors(gguf: &Gguf) -> Result<()> {
    for layer in 0..LAYER_COUNT {
        validate_block_tensors(gguf, layer)?;
    }
    for layer in DEEPSTACK_LAYERS {
        validate_deepstack_tensors(gguf, layer)?;
    }
    expect_tensor(gguf, "mm.0.bias", &[FFN_WIDTH], TensorType::F32)?;
    expect_tensor(
        gguf,
        "mm.0.weight",
        &[FFN_WIDTH, FFN_WIDTH],
        TensorType::Q8_0,
    )?;
    expect_tensor(gguf, "mm.2.bias", &[PROJECTED_WIDTH], TensorType::F32)?;
    expect_tensor(
        gguf,
        "mm.2.weight",
        &[FFN_WIDTH, PROJECTED_WIDTH],
        TensorType::Q8_0,
    )?;
    expect_tensor(gguf, "v.post_ln.bias", &[EMBEDDING_WIDTH], TensorType::F32)?;
    expect_tensor(
        gguf,
        "v.post_ln.weight",
        &[EMBEDDING_WIDTH],
        TensorType::F32,
    )?;
    expect_tensor(
        gguf,
        "v.patch_embd.bias",
        &[EMBEDDING_WIDTH],
        TensorType::F32,
    )?;
    expect_tensor(
        gguf,
        "v.patch_embd.weight",
        &[PATCH_SIZE, PATCH_SIZE, CHANNELS, EMBEDDING_WIDTH],
        TensorType::F32,
    )?;
    expect_tensor(
        gguf,
        "v.patch_embd.weight.1",
        &[PATCH_SIZE, PATCH_SIZE, CHANNELS, EMBEDDING_WIDTH],
        TensorType::F32,
    )?;
    expect_tensor(
        gguf,
        "v.position_embd.weight",
        &[EMBEDDING_WIDTH, POSITION_SIDE * POSITION_SIDE],
        TensorType::F32,
    )?;
    Ok(())
}

fn validate_block_tensors(gguf: &Gguf, layer: usize) -> Result<()> {
    let prefix = format!("v.blk.{layer}");
    expect_tensor(
        gguf,
        &format!("{prefix}.attn_out.bias"),
        &[EMBEDDING_WIDTH],
        TensorType::F32,
    )?;
    expect_tensor(
        gguf,
        &format!("{prefix}.attn_out.weight"),
        &[EMBEDDING_WIDTH, EMBEDDING_WIDTH],
        TensorType::Q8_0,
    )?;
    expect_tensor(
        gguf,
        &format!("{prefix}.attn_qkv.bias"),
        &[QKV_WIDTH],
        TensorType::F32,
    )?;
    expect_tensor(
        gguf,
        &format!("{prefix}.attn_qkv.weight"),
        &[EMBEDDING_WIDTH, QKV_WIDTH],
        TensorType::Q8_0,
    )?;
    expect_tensor(
        gguf,
        &format!("{prefix}.ffn_up.bias"),
        &[FFN_WIDTH],
        TensorType::F32,
    )?;
    expect_tensor(
        gguf,
        &format!("{prefix}.ffn_up.weight"),
        &[EMBEDDING_WIDTH, FFN_WIDTH],
        TensorType::Q8_0,
    )?;
    expect_tensor(
        gguf,
        &format!("{prefix}.ffn_down.bias"),
        &[EMBEDDING_WIDTH],
        TensorType::F32,
    )?;
    expect_tensor(
        gguf,
        &format!("{prefix}.ffn_down.weight"),
        &[FFN_WIDTH, EMBEDDING_WIDTH],
        TensorType::Q8_0,
    )?;
    expect_tensor(
        gguf,
        &format!("{prefix}.ln1.bias"),
        &[EMBEDDING_WIDTH],
        TensorType::F32,
    )?;
    expect_tensor(
        gguf,
        &format!("{prefix}.ln1.weight"),
        &[EMBEDDING_WIDTH],
        TensorType::F32,
    )?;
    expect_tensor(
        gguf,
        &format!("{prefix}.ln2.bias"),
        &[EMBEDDING_WIDTH],
        TensorType::F32,
    )?;
    expect_tensor(
        gguf,
        &format!("{prefix}.ln2.weight"),
        &[EMBEDDING_WIDTH],
        TensorType::F32,
    )?;
    Ok(())
}

fn validate_deepstack_tensors(gguf: &Gguf, layer: usize) -> Result<()> {
    let prefix = format!("v.deepstack.{layer}");
    expect_tensor(
        gguf,
        &format!("{prefix}.fc1.bias"),
        &[FFN_WIDTH],
        TensorType::F32,
    )?;
    expect_tensor(
        gguf,
        &format!("{prefix}.fc1.weight"),
        &[FFN_WIDTH, FFN_WIDTH],
        TensorType::Q8_0,
    )?;
    expect_tensor(
        gguf,
        &format!("{prefix}.fc2.bias"),
        &[PROJECTED_WIDTH],
        TensorType::F32,
    )?;
    expect_tensor(
        gguf,
        &format!("{prefix}.fc2.weight"),
        &[FFN_WIDTH, PROJECTED_WIDTH],
        TensorType::Q8_0,
    )?;
    expect_tensor(
        gguf,
        &format!("{prefix}.norm.bias"),
        &[FFN_WIDTH],
        TensorType::F32,
    )?;
    expect_tensor(
        gguf,
        &format!("{prefix}.norm.weight"),
        &[FFN_WIDTH],
        TensorType::F32,
    )?;
    Ok(())
}

fn expect_tensor(gguf: &Gguf, name: &str, dimensions: &[usize], kind: TensorType) -> Result<()> {
    let tensor: Tensor<'_> = gguf.tensor(name)?;
    ensure!(
        tensor.dimensions() == dimensions,
        "tensor `{name}` has dimensions {:?}, expected {dimensions:?}",
        tensor.dimensions()
    );
    ensure!(
        tensor.tensor_type() == kind,
        "tensor `{name}` is {:?}, expected {kind:?}",
        tensor.tensor_type()
    );
    match kind {
        TensorType::F32 => {
            tensor.f32_slice()?;
        }
        TensorType::Q8_0 => {
            tensor.q8_row_size()?;
        }
        TensorType::Q4K | TensorType::Q6K => {
            tensor.quantized_row_size()?;
        }
    }
    Ok(())
}
