// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (c) 2026 Jarkko Sakkinen

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, ensure};
use rayon::prelude::*;

use super::gguf::{Gguf, TensorType};
use super::kernels::{
    Fp16AttentionKernel, dequantize_q4_k_row, dequantize_q6_k_row, dim_to_f32, f32_to_fp16,
    matrix_matrix, matrix_matrix_pair, matrix_matrix_triple, matrix_vector_argmax, rms_norm, silu,
    softmax, vector_add, vector_multiply,
};
use super::tokenizer::{Tokenizer, Utf8Decoder};
use super::vision::VisionEmbedding;

const LAYER_COUNT: usize = 28;
const EMBEDDING_SIZE: usize = 2_048;
const FEED_FORWARD_SIZE: usize = 6_144;
const QUERY_HEAD_COUNT: usize = 16;
const KEY_VALUE_HEAD_COUNT: usize = 8;
const HEAD_SIZE: usize = 128;
const KEY_VALUE_SIZE: usize = KEY_VALUE_HEAD_COUNT * HEAD_SIZE;
const QUERY_GROUP_SIZE: usize = QUERY_HEAD_COUNT / KEY_VALUE_HEAD_COUNT;
const DEEPSTACK_LAYER_COUNT: usize = 3;
const IMAGE_EMBEDDING_SIZE: usize = EMBEDDING_SIZE * (DEEPSTACK_LAYER_COUNT + 1);
const RMS_NORM_EPSILON: f32 = 1.0e-6;
const ROPE_BASE: f32 = 5_000_000.0;
const ROPE_SECTIONS: [u32; 3] = [24, 20, 20];
const PREFILL_BATCH_SIZE: usize = 32;

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const VISION_START: &str = "<|vision_start|>";
const VISION_END: &str = "<|vision_end|>";

/// Text and timing information from one greedy generation request.
#[derive(Debug)]
pub struct Generation {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_duration: Duration,
    pub decode_duration: Duration,
}

/// Loaded, immutable Qwen3-VL-2B text decoder.
pub struct TextModel {
    gguf: Gguf,
    tokenizer: Tokenizer,
    layers: Vec<TextLayerNames>,
}

struct TextLayerNames {
    attention_norm: String,
    query: String,
    key: String,
    value: String,
    query_norm: String,
    key_norm: String,
    attention_output: String,
    feed_forward_norm: String,
    feed_forward_gate: String,
    feed_forward_up: String,
    feed_forward_down: String,
}

impl TextLayerNames {
    fn new(layer: usize) -> Self {
        let prefix = format!("blk.{layer}");
        Self {
            attention_norm: format!("{prefix}.attn_norm.weight"),
            query: format!("{prefix}.attn_q.weight"),
            key: format!("{prefix}.attn_k.weight"),
            value: format!("{prefix}.attn_v.weight"),
            query_norm: format!("{prefix}.attn_q_norm.weight"),
            key_norm: format!("{prefix}.attn_k_norm.weight"),
            attention_output: format!("{prefix}.attn_output.weight"),
            feed_forward_norm: format!("{prefix}.ffn_norm.weight"),
            feed_forward_gate: format!("{prefix}.ffn_gate.weight"),
            feed_forward_up: format!("{prefix}.ffn_up.weight"),
            feed_forward_down: format!("{prefix}.ffn_down.weight"),
        }
    }
}

impl TextModel {
    /// Load and fully validate the pinned Qwen3-VL-2B `Q8_0` decoder.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let gguf = Gguf::load(path).context("parse text GGUF")?;
        validate_metadata(&gguf).context("validate text GGUF metadata")?;
        validate_tensors(&gguf).context("validate text GGUF tensors")?;

        let tokenizer = Tokenizer::from_gguf(&gguf).context("load text tokenizer")?;
        validate_special_tokens(&tokenizer).context("validate text special tokens")?;
        let layers = (0..LAYER_COUNT).map(TextLayerNames::new).collect();
        Ok(Self {
            gguf,
            tokenizer,
            layers,
        })
    }

    /// Generate greedily from the fixed `ReadSeek` one-image chat template.
    pub fn generate(
        &self,
        prompt: &str,
        image: &VisionEmbedding,
        max_new_tokens: usize,
    ) -> Result<Generation> {
        let image_tokens = validate_image(image)?;
        let prefix_tokens = self
            .tokenizer
            .encode(&format!("{IM_START}user\n{VISION_START}"), true)
            .context("tokenize chat prefix")?;
        let suffix_tokens = self
            .tokenizer
            .encode(
                &format!("{VISION_END}{prompt}{IM_END}\n{IM_START}assistant\n"),
                true,
            )
            .context("tokenize chat suffix")?;
        let prompt_tokens = prefix_tokens
            .len()
            .checked_add(image_tokens)
            .and_then(|count| count.checked_add(suffix_tokens.len()))
            .context("prompt token count overflow")?;
        let cache_tokens = prompt_tokens
            .checked_add(max_new_tokens)
            .context("key/value cache token count overflow")?;
        let mut cache = KvCache::new(cache_tokens)?;
        let mut scratch = TextAttentionScratch::new(cache_tokens)?;
        let mut position = 0_usize;

        let prefill_started = Instant::now();
        let after_prefix =
            self.prefill_text(&prefix_tokens, &mut cache, &mut scratch, &mut position)?;
        let after_image = self.prefill_image(image, &mut cache, &mut scratch, &mut position)?;
        let after_suffix =
            self.prefill_text(&suffix_tokens, &mut cache, &mut scratch, &mut position)?;
        let hidden = after_suffix
            .or(after_image)
            .or(after_prefix)
            .context("chat prompt produced no decoder input")?;
        let token = self.greedy_token(&hidden)?;
        let prefill_duration = prefill_started.elapsed();

        let decode_started = Instant::now();
        let (text, generated_tokens) = self.decode(
            token,
            max_new_tokens,
            &mut cache,
            &mut scratch,
            &mut position,
        )?;
        let decode_duration = decode_started.elapsed();

        Ok(Generation {
            text,
            prompt_tokens,
            generated_tokens,
            prefill_duration,
            decode_duration,
        })
    }

    /// Feed `tokens` through the decoder, advancing `position` by one per token.
    fn prefill_text(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        scratch: &mut TextAttentionScratch,
        position: &mut usize,
    ) -> Result<Option<Vec<f32>>> {
        let mut last_hidden = None;
        for chunk in tokens.chunks(PREFILL_BATCH_SIZE) {
            let mut input = Vec::with_capacity(chunk.len() * EMBEDDING_SIZE);
            let mut positions = Vec::with_capacity(chunk.len());
            for (offset, token) in chunk.iter().enumerate() {
                input.extend(self.token_embedding(*token)?);
                let current = (*position)
                    .checked_add(offset)
                    .context("text position overflow")?;
                positions.push(text_position(current)?);
            }
            let masses = vec![1; chunk.len()];
            let hidden = self.forward_batch(&input, &positions, None, &masses, cache, scratch)?;
            last_hidden = hidden
                .chunks_exact(EMBEDDING_SIZE)
                .last()
                .map(<[f32]>::to_vec);
            *position = (*position)
                .checked_add(chunk.len())
                .context("text position overflow")?;
        }
        Ok(last_hidden)
    }

    /// Project image patch embeddings through the decoder, advancing `position`
    /// by the larger grid dimension at the end.
    fn prefill_image(
        &self,
        image: &VisionEmbedding,
        cache: &mut KvCache,
        scratch: &mut TextAttentionScratch,
        position: &mut usize,
    ) -> Result<Option<Vec<f32>>> {
        let mut last_hidden = None;
        for start in (0..image.token_count).step_by(PREFILL_BATCH_SIZE) {
            let end = (start + PREFILL_BATCH_SIZE).min(image.token_count);
            let mut input = Vec::with_capacity((end - start) * EMBEDDING_SIZE);
            let mut deepstack =
                Vec::with_capacity((end - start) * DEEPSTACK_LAYER_COUNT * EMBEDDING_SIZE);
            let mut positions = Vec::with_capacity(end - start);
            for index in start..end {
                let row_start = index * IMAGE_EMBEDDING_SIZE;
                let row = &image.values[row_start..row_start + IMAGE_EMBEDDING_SIZE];
                input.extend_from_slice(&row[..EMBEDDING_SIZE]);
                deepstack.extend_from_slice(&row[EMBEDDING_SIZE..]);
                positions.push(image_position(
                    *position,
                    image.positions[index],
                    image.grid_width,
                    image.grid_height,
                )?);
            }
            let hidden = self.forward_batch(
                &input,
                &positions,
                Some(&deepstack),
                &image.masses[start..end],
                cache,
                scratch,
            )?;
            last_hidden = hidden
                .chunks_exact(EMBEDDING_SIZE)
                .last()
                .map(<[f32]>::to_vec);
        }
        *position = (*position)
            .checked_add(image.grid_width.max(image.grid_height))
            .context("image position overflow")?;
        Ok(last_hidden)
    }

    /// Greedily decode up to `max_new_tokens`, threading the shared cache.
    fn decode(
        &self,
        mut token: u32,
        max_new_tokens: usize,
        cache: &mut KvCache,
        scratch: &mut TextAttentionScratch,
        position: &mut usize,
    ) -> Result<(String, usize)> {
        let mut decoder = Utf8Decoder::new();
        let mut json = JsonObjectTracker::default();
        let mut text = String::new();
        let mut generated_tokens = 0_usize;

        while generated_tokens < max_new_tokens {
            if self.tokenizer.is_eos(token) {
                break;
            }
            generated_tokens += 1;

            let mut json_complete = false;
            if let Some(decoded) = decoder.push(self.tokenizer.token_piece(token)?)? {
                if let Some(end) = json.push(&decoded) {
                    text.push_str(&decoded[..end]);
                    json_complete = true;
                } else {
                    text.push_str(&decoded);
                }
            }
            if json_complete || generated_tokens == max_new_tokens {
                break;
            }

            let embedding = self.token_embedding(token)?;
            let current = text_position(*position)?;
            let hidden = self.forward_token(&embedding, current, None, 1, cache, scratch)?;
            *position = (*position)
                .checked_add(1)
                .context("text position overflow")?;
            token = self.greedy_token(&hidden)?;
        }

        text.push_str(&decoder.finish()?);
        Ok((text, generated_tokens))
    }

    fn token_embedding(&self, token: u32) -> Result<Vec<f32>> {
        let tensor = self.gguf.tensor("token_embd.weight")?;
        let row = tensor
            .quantized_row_bytes(token as usize)
            .with_context(|| format!("read token embedding {token}"))?;
        let mut embedding = vec![0.0; EMBEDDING_SIZE];
        match tensor.tensor_type() {
            TensorType::Q4K => dequantize_q4_k_row(row, &mut embedding)?,
            TensorType::Q6K => dequantize_q6_k_row(row, &mut embedding)?,
            kind => anyhow::bail!("token embedding uses unsupported tensor type {kind:?}"),
        }
        Ok(embedding)
    }

    fn forward_token(
        &self,
        input: &[f32],
        position: Position,
        deepstack: Option<&[f32]>,
        mass: u32,
        cache: &mut KvCache,
        attention_scratch: &mut TextAttentionScratch,
    ) -> Result<Vec<f32>> {
        self.forward_batch(
            input,
            std::slice::from_ref(&position),
            deepstack,
            std::slice::from_ref(&mass),
            cache,
            attention_scratch,
        )
    }

    fn forward_batch(
        &self,
        input: &[f32],
        positions: &[Position],
        deepstack: Option<&[f32]>,
        masses: &[u32],
        cache: &mut KvCache,
        attention_scratch: &mut TextAttentionScratch,
    ) -> Result<Vec<f32>> {
        let token_count = validate_batch(input, positions, deepstack, masses)?;
        let input_size = token_count * EMBEDDING_SIZE;
        let ropes = positions
            .iter()
            .copied()
            .map(ImRope::new)
            .collect::<Vec<_>>();

        let mut hidden = input.to_vec();
        for (layer, names) in self.layers.iter().enumerate() {
            let attention_norm = self.gguf.tensor(&names.attention_norm)?;
            let normalized = rms_norm(
                &hidden,
                EMBEDDING_SIZE,
                attention_norm.f32_slice()?,
                RMS_NORM_EPSILON,
            )?;

            let (mut query, mut key, value) = matrix_matrix_triple(
                &self.gguf.tensor(&names.query)?,
                &self.gguf.tensor(&names.key)?,
                &self.gguf.tensor(&names.value)?,
                &normalized,
                token_count,
            )?;

            let query_norm = self.gguf.tensor(&names.query_norm)?;
            query = rms_norm(&query, HEAD_SIZE, query_norm.f32_slice()?, RMS_NORM_EPSILON)?;
            let key_norm = self.gguf.tensor(&names.key_norm)?;
            key = rms_norm(&key, HEAD_SIZE, key_norm.f32_slice()?, RMS_NORM_EPSILON)?;
            for ((query, key), rope) in query
                .chunks_exact_mut(EMBEDDING_SIZE)
                .zip(key.chunks_exact_mut(KEY_VALUE_SIZE))
                .zip(&ropes)
            {
                apply_im_rope(query, rope)?;
                apply_im_rope(key, rope)?;
            }

            let cached_tokens = cache.layers[layer].token_count();
            for ((key, value), mass) in key
                .chunks_exact(KEY_VALUE_SIZE)
                .zip(value.chunks_exact(KEY_VALUE_SIZE))
                .zip(masses)
            {
                cache.layers[layer].append(key, value, *mass)?;
            }
            let mut attention = Vec::with_capacity(input_size);
            for (token, query) in query.chunks_exact(EMBEDDING_SIZE).enumerate() {
                causal_gqa(
                    query,
                    &cache.layers[layer],
                    cached_tokens + token + 1,
                    attention_scratch,
                )?;
                attention.extend_from_slice(&attention_scratch.output);
            }
            let projected = matrix_matrix(
                &self.gguf.tensor(&names.attention_output)?,
                &attention,
                token_count,
            )?;
            vector_add(&mut hidden, &projected)?;

            let feed_forward_norm = self.gguf.tensor(&names.feed_forward_norm)?;
            let normalized = rms_norm(
                &hidden,
                EMBEDDING_SIZE,
                feed_forward_norm.f32_slice()?,
                RMS_NORM_EPSILON,
            )?;
            let (mut gate, up) = matrix_matrix_pair(
                &self.gguf.tensor(&names.feed_forward_gate)?,
                &self.gguf.tensor(&names.feed_forward_up)?,
                &normalized,
                token_count,
            )?;
            silu(&mut gate);
            vector_multiply(&mut gate, &up)?;
            let down = matrix_matrix(
                &self.gguf.tensor(&names.feed_forward_down)?,
                &gate,
                token_count,
            )?;
            vector_add(&mut hidden, &down)?;

            if layer < DEEPSTACK_LAYER_COUNT
                && let Some(deepstack) = deepstack
            {
                add_deepstack(&mut hidden, deepstack, layer)?;
            }
        }

        let output_norm = self.gguf.tensor("output_norm.weight")?;
        rms_norm(
            &hidden,
            EMBEDDING_SIZE,
            output_norm.f32_slice()?,
            RMS_NORM_EPSILON,
        )
    }

    fn greedy_token(&self, hidden: &[f32]) -> Result<u32> {
        let index = matrix_vector_argmax(&self.gguf.tensor("token_embd.weight")?, hidden)
            .context("compute tied-embedding token")?;
        u32::try_from(index).context("sampled token ID exceeds u32")
    }
}

fn validate_batch(
    input: &[f32],
    positions: &[Position],
    deepstack: Option<&[f32]>,
    masses: &[u32],
) -> Result<usize> {
    let token_count = positions.len();
    ensure!(
        token_count != 0 && token_count <= PREFILL_BATCH_SIZE,
        "decoder batch has {token_count} tokens, expected 1..={PREFILL_BATCH_SIZE}"
    );
    let input_size = token_count * EMBEDDING_SIZE;
    ensure!(
        input.len() == input_size,
        "decoder input has {} values, expected {input_size}",
        input.len()
    );
    ensure!(
        masses.len() == token_count,
        "decoder batch has {} masses, expected {token_count}",
        masses.len()
    );
    if let Some(deepstack) = deepstack {
        let deepstack_size = token_count * DEEPSTACK_LAYER_COUNT * EMBEDDING_SIZE;
        ensure!(
            deepstack.len() == deepstack_size,
            "DeepStack input has {} values, expected {deepstack_size}",
            deepstack.len(),
        );
    }
    Ok(token_count)
}

fn add_deepstack(hidden: &mut [f32], deepstack: &[f32], layer: usize) -> Result<()> {
    for (hidden, deepstack) in hidden
        .chunks_exact_mut(EMBEDDING_SIZE)
        .zip(deepstack.chunks_exact(DEEPSTACK_LAYER_COUNT * EMBEDDING_SIZE))
    {
        let start = layer * EMBEDDING_SIZE;
        vector_add(hidden, &deepstack[start..start + EMBEDDING_SIZE])?;
    }
    Ok(())
}

type Position = [u32; 4];

fn text_position(position: usize) -> Result<Position> {
    let position = u32::try_from(position).context("text position exceeds u32")?;
    Ok([position; 4])
}

fn image_position(
    scalar: usize,
    position: [usize; 2],
    grid_width: usize,
    grid_height: usize,
) -> Result<Position> {
    ensure!(grid_width != 0 && grid_height != 0, "image grid is empty");
    let [row, column] = position;
    ensure!(
        row < grid_height && column < grid_width,
        "image token position [{row}, {column}] is outside the grid"
    );

    let temporal = u32::try_from(scalar).context("image temporal position exceeds u32")?;
    let height = scalar
        .checked_add(row)
        .context("image height position overflow")?;
    let width = scalar
        .checked_add(column)
        .context("image width position overflow")?;
    Ok([
        temporal,
        u32::try_from(height).context("image height position exceeds u32")?,
        u32::try_from(width).context("image width position exceeds u32")?,
        0,
    ])
}

struct ImRope {
    cosine: [f32; HEAD_SIZE / 2],
    sine: [f32; HEAD_SIZE / 2],
}
/// `u32 -> f32` for `RoPE` positions. Positions are token/grid offsets that may
/// exceed `u16`, so the exact-dimension helper does not apply; the checked
/// `to_f32()` is total for `u32` (every value is representable in `f32`)
/// while keeping the lossiness explicit at the call site.
use num_traits::ToPrimitive;

fn position_to_f32(position: u32) -> f32 {
    position.to_f32().unwrap_or(0.0)
}
impl ImRope {
    fn new(position: Position) -> Self {
        let temporal = position_to_f32(position[0]);
        let height = position_to_f32(position[1]);
        let width = position_to_f32(position[2]);
        let mut rope = Self {
            cosine: [0.0; HEAD_SIZE / 2],
            sine: [0.0; HEAD_SIZE / 2],
        };

        for pair in 0..HEAD_SIZE / 2 {
            let coordinate = if pair % 3 == 1 && pair < 3 * ROPE_SECTIONS[1] as usize {
                height
            } else if pair % 3 == 2 && pair < 3 * ROPE_SECTIONS[2] as usize {
                width
            } else {
                temporal
            };
            let frequency = ROPE_BASE.powf(-(dim_to_f32(2 * pair) / dim_to_f32(HEAD_SIZE)));
            let angle = coordinate * frequency;
            rope.cosine[pair] = angle.cos();
            rope.sine[pair] = angle.sin();
        }
        rope
    }
}

fn apply_im_rope(values: &mut [f32], rope: &ImRope) -> Result<()> {
    ensure!(
        values.len().is_multiple_of(HEAD_SIZE),
        "RoPE input has {} values, not a multiple of {HEAD_SIZE}",
        values.len()
    );
    values.par_chunks_mut(HEAD_SIZE).for_each(|head| {
        let (first, second) = head.split_at_mut(HEAD_SIZE / 2);
        for pair in 0..HEAD_SIZE / 2 {
            let left = first[pair];
            let right = second[pair];
            first[pair] = left * rope.cosine[pair] - right * rope.sine[pair];
            second[pair] = left * rope.sine[pair] + right * rope.cosine[pair];
        }
    });
    Ok(())
}

#[derive(Default)]
struct LayerCache {
    keys: Vec<u16>,
    values: Vec<u16>,
    log_masses: Vec<f32>,
}

impl LayerCache {
    fn reserve_tokens(&mut self, token_count: usize) -> Result<()> {
        let values = token_count
            .checked_mul(KEY_VALUE_SIZE)
            .context("F16 cache capacity overflow")?;
        self.keys
            .try_reserve_exact(values)
            .context("reserve F16 key cache")?;
        self.values
            .try_reserve_exact(values)
            .context("reserve F16 value cache")?;
        self.log_masses
            .try_reserve_exact(token_count)
            .context("reserve attention mass cache")?;
        Ok(())
    }
    fn append(&mut self, key: &[f32], value: &[f32], mass: u32) -> Result<()> {
        ensure!(
            key.len() == KEY_VALUE_SIZE,
            "cache key has {} values, expected {KEY_VALUE_SIZE}",
            key.len()
        );
        ensure!(
            value.len() == KEY_VALUE_SIZE,
            "cache value has {} values, expected {KEY_VALUE_SIZE}",
            value.len()
        );
        ensure!(
            self.keys.len() == self.values.len() && self.token_count() == self.log_masses.len(),
            "key, value, and mass cache lengths differ"
        );
        ensure!(mass != 0, "attention token mass is zero");
        self.keys
            .try_reserve(KEY_VALUE_SIZE)
            .context("grow F16 key cache")?;
        self.values
            .try_reserve(KEY_VALUE_SIZE)
            .context("grow F16 value cache")?;
        self.log_masses
            .try_reserve(1)
            .context("grow attention mass cache")?;
        self.keys.extend(key.iter().copied().map(f32_to_fp16));
        self.values.extend(value.iter().copied().map(f32_to_fp16));
        self.log_masses
            .push((f32::from(u16::try_from(mass).expect("attention mass fits u16"))).ln());
        Ok(())
    }

    fn token_count(&self) -> usize {
        self.keys.len() / KEY_VALUE_SIZE
    }
}

struct KvCache {
    layers: Vec<LayerCache>,
}

impl KvCache {
    fn new(prompt_tokens: usize) -> Result<Self> {
        let mut layers: Vec<_> = (0..LAYER_COUNT).map(|_| LayerCache::default()).collect();
        for layer in &mut layers {
            layer.reserve_tokens(prompt_tokens)?;
        }
        Ok(Self { layers })
    }
}

#[derive(Default)]
struct TextAttentionScratch {
    output: Vec<f32>,
    scores: Vec<f32>,
}

impl TextAttentionScratch {
    fn new(token_capacity: usize) -> Result<Self> {
        let score_capacity = QUERY_HEAD_COUNT
            .checked_mul(token_capacity)
            .context("attention score capacity overflow")?;
        let mut scores = Vec::new();
        scores
            .try_reserve_exact(score_capacity)
            .context("reserve attention score workspace")?;
        Ok(Self {
            output: vec![0.0; EMBEDDING_SIZE],
            scores,
        })
    }
}

fn causal_gqa(
    query: &[f32],
    cache: &LayerCache,
    visible_tokens: usize,
    scratch: &mut TextAttentionScratch,
) -> Result<()> {
    ensure!(
        query.len() == EMBEDDING_SIZE,
        "attention query has {} values, expected {EMBEDDING_SIZE}",
        query.len()
    );
    ensure!(
        cache.keys.len() == cache.values.len()
            && cache.keys.len().is_multiple_of(KEY_VALUE_SIZE)
            && cache.token_count() == cache.log_masses.len(),
        "invalid key/value cache shape"
    );
    ensure!(
        visible_tokens != 0 && visible_tokens <= cache.token_count(),
        "attention visibility is {visible_tokens}, but cache has {} tokens",
        cache.token_count()
    );
    let score_count = QUERY_HEAD_COUNT
        .checked_mul(visible_tokens)
        .context("attention score count overflow")?;
    scratch.output.resize(EMBEDDING_SIZE, 0.0);
    scratch.output.fill(0.0);
    scratch.scores.resize(score_count, 0.0);
    let scale = dim_to_f32(HEAD_SIZE).sqrt().recip();
    let kernel = Fp16AttentionKernel::detect();

    scratch
        .output
        .par_chunks_mut(QUERY_GROUP_SIZE * HEAD_SIZE)
        .zip(
            scratch
                .scores
                .par_chunks_mut(QUERY_GROUP_SIZE * visible_tokens),
        )
        .enumerate()
        .for_each(|(key_value_head, (output_heads, scores))| {
            let (output_left, output_right) = output_heads.split_at_mut(HEAD_SIZE);
            let (weights_left, weights_right) = scores.split_at_mut(visible_tokens);
            let query_start = key_value_head * QUERY_GROUP_SIZE * HEAD_SIZE;
            let query_left = &query[query_start..query_start + HEAD_SIZE];
            let query_right = &query[query_start + HEAD_SIZE..query_start + 2 * HEAD_SIZE];
            let head_start = key_value_head * HEAD_SIZE;
            for (((keys, log_mass), left_weight), right_weight) in cache
                .keys
                .chunks_exact(KEY_VALUE_SIZE)
                .zip(&cache.log_masses)
                .take(visible_tokens)
                .zip(weights_left.iter_mut())
                .zip(weights_right.iter_mut())
            {
                let keys = &keys[head_start..head_start + HEAD_SIZE];
                let (dot_left, dot_right) = kernel.dot_pair(keys, query_left, query_right);
                *left_weight = dot_left * scale + log_mass;
                *right_weight = dot_right * scale + log_mass;
            }
            softmax(weights_left);
            softmax(weights_right);
            for ((values, left_weight), right_weight) in cache
                .values
                .chunks_exact(KEY_VALUE_SIZE)
                .take(visible_tokens)
                .zip(weights_left.iter())
                .zip(weights_right.iter())
            {
                let values = &values[head_start..head_start + HEAD_SIZE];
                kernel.accumulate_pair(
                    values,
                    *left_weight,
                    *right_weight,
                    output_left,
                    output_right,
                );
            }
        });
    ensure!(
        scratch.output.iter().all(|value| value.is_finite()),
        "attention produced a non-finite value"
    );
    Ok(())
}
fn validate_image(image: &VisionEmbedding) -> Result<usize> {
    ensure!(
        image.grid_width != 0 && image.grid_height != 0,
        "image decoder grid must be nonempty"
    );
    let grid_tokens = image
        .grid_width
        .checked_mul(image.grid_height)
        .context("image decoder grid size overflow")?;
    ensure!(
        image.original_token_count == grid_tokens,
        "image original token count does not match its grid"
    );
    ensure!(
        image.token_count != 0 && image.token_count <= grid_tokens,
        "image reduced token count is invalid"
    );
    ensure!(
        image.positions.len() == image.token_count && image.masses.len() == image.token_count,
        "image token metadata length differs"
    );
    let expected = image
        .token_count
        .checked_mul(IMAGE_EMBEDDING_SIZE)
        .context("image embedding size overflow")?;
    ensure!(
        image.values.len() == expected,
        "image embedding has {} values, expected {expected}",
        image.values.len()
    );
    ensure!(
        image.values.iter().all(|value| value.is_finite()),
        "image embedding contains a non-finite value"
    );
    ensure!(
        image.masses.iter().all(|mass| *mass != 0),
        "image contains a zero-mass token"
    );
    ensure!(
        image
            .positions
            .iter()
            .all(|[row, column]| { *row < image.grid_height && *column < image.grid_width }),
        "image token position is outside the grid"
    );
    ensure!(
        image.positions.windows(2).all(|positions| {
            positions[0][0] * image.grid_width + positions[0][1]
                < positions[1][0] * image.grid_width + positions[1][1]
        }),
        "image token positions are not strictly row-major"
    );
    Ok(image.token_count)
}

fn validate_metadata(gguf: &Gguf) -> Result<()> {
    ensure!(
        gguf.architecture() == "qwen3vl",
        "model architecture is `{}`, expected `qwen3vl`",
        gguf.architecture()
    );
    ensure!(
        gguf.u32("general.file_type")? == 15,
        "GGUF file type must be Q4_K_M"
    );
    validate_u32(gguf, "qwen3vl.block_count", LAYER_COUNT)?;
    validate_u32(gguf, "qwen3vl.embedding_length", EMBEDDING_SIZE)?;
    validate_u32(gguf, "qwen3vl.feed_forward_length", FEED_FORWARD_SIZE)?;
    validate_u32(gguf, "qwen3vl.attention.head_count", QUERY_HEAD_COUNT)?;
    validate_u32(
        gguf,
        "qwen3vl.attention.head_count_kv",
        KEY_VALUE_HEAD_COUNT,
    )?;
    validate_u32(gguf, "qwen3vl.attention.key_length", HEAD_SIZE)?;
    validate_u32(gguf, "qwen3vl.attention.value_length", HEAD_SIZE)?;
    validate_u32(gguf, "qwen3vl.n_deepstack_layers", DEEPSTACK_LAYER_COUNT)?;

    let rope_base = gguf.f32("qwen3vl.rope.freq_base")?;
    ensure!(
        rope_base.to_bits() == ROPE_BASE.to_bits(),
        "qwen3vl.rope.freq_base is {rope_base}, expected {ROPE_BASE}"
    );
    let epsilon = gguf.f32("qwen3vl.attention.layer_norm_rms_epsilon")?;
    ensure!(
        epsilon.to_bits() == RMS_NORM_EPSILON.to_bits(),
        "qwen3vl.attention.layer_norm_rms_epsilon is {epsilon}, expected {RMS_NORM_EPSILON}"
    );
    let sections = gguf.u32_array("qwen3vl.rope.dimension_sections")?;
    let canonical = sections.as_ref() == ROPE_SECTIONS;
    let zero_padded = sections.as_ref() == [24, 20, 20, 0];
    ensure!(
        canonical || zero_padded,
        "qwen3vl.rope.dimension_sections is {sections:?}, expected {ROPE_SECTIONS:?}"
    );
    Ok(())
}

fn validate_u32(gguf: &Gguf, key: &str, expected: usize) -> Result<()> {
    let value = usize::try_from(gguf.u32(key)?).expect("metadata value fits usize");
    ensure!(value == expected, "{key} is {value}, expected {expected}");
    Ok(())
}

fn validate_tensors(gguf: &Gguf) -> Result<()> {
    validate_tensor(
        gguf,
        "token_embd.weight",
        &[EMBEDDING_SIZE, tokenizer_vocabulary_size(gguf)?],
        TensorType::Q6K,
    )?;
    validate_tensor(
        gguf,
        "output_norm.weight",
        &[EMBEDDING_SIZE],
        TensorType::F32,
    )?;

    for layer in 0..LAYER_COUNT {
        validate_tensor(
            gguf,
            &format!("blk.{layer}.attn_norm.weight"),
            &[EMBEDDING_SIZE],
            TensorType::F32,
        )?;
        validate_tensor(
            gguf,
            &format!("blk.{layer}.attn_q.weight"),
            &[EMBEDDING_SIZE, EMBEDDING_SIZE],
            TensorType::Q4K,
        )?;
        validate_tensor(
            gguf,
            &format!("blk.{layer}.attn_k.weight"),
            &[EMBEDDING_SIZE, KEY_VALUE_SIZE],
            TensorType::Q4K,
        )?;
        validate_tensor(
            gguf,
            &format!("blk.{layer}.attn_v.weight"),
            &[EMBEDDING_SIZE, KEY_VALUE_SIZE],
            mixed_weight_kind(layer),
        )?;
        validate_tensor(
            gguf,
            &format!("blk.{layer}.attn_output.weight"),
            &[EMBEDDING_SIZE, EMBEDDING_SIZE],
            TensorType::Q4K,
        )?;
        validate_tensor(
            gguf,
            &format!("blk.{layer}.attn_q_norm.weight"),
            &[HEAD_SIZE],
            TensorType::F32,
        )?;
        validate_tensor(
            gguf,
            &format!("blk.{layer}.attn_k_norm.weight"),
            &[HEAD_SIZE],
            TensorType::F32,
        )?;
        validate_tensor(
            gguf,
            &format!("blk.{layer}.ffn_norm.weight"),
            &[EMBEDDING_SIZE],
            TensorType::F32,
        )?;
        validate_tensor(
            gguf,
            &format!("blk.{layer}.ffn_gate.weight"),
            &[EMBEDDING_SIZE, FEED_FORWARD_SIZE],
            TensorType::Q4K,
        )?;
        validate_tensor(
            gguf,
            &format!("blk.{layer}.ffn_up.weight"),
            &[EMBEDDING_SIZE, FEED_FORWARD_SIZE],
            TensorType::Q4K,
        )?;
        validate_tensor(
            gguf,
            &format!("blk.{layer}.ffn_down.weight"),
            &[FEED_FORWARD_SIZE, EMBEDDING_SIZE],
            mixed_weight_kind(layer),
        )?;
    }
    Ok(())
}

fn mixed_weight_kind(layer: usize) -> TensorType {
    if matches!(
        layer,
        0 | 1 | 2 | 5 | 8 | 11 | 14 | 17 | 20 | 23 | 24 | 25 | 26 | 27
    ) {
        TensorType::Q6K
    } else {
        TensorType::Q4K
    }
}

fn tokenizer_vocabulary_size(gguf: &Gguf) -> Result<usize> {
    Ok(gguf.string_array("tokenizer.ggml.tokens")?.len())
}

fn validate_tensor(gguf: &Gguf, name: &str, dimensions: &[usize], kind: TensorType) -> Result<()> {
    let tensor = gguf.tensor(name)?;
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
        TensorType::Q8_0 | TensorType::Q4K | TensorType::Q6K => {
            tensor.quantized_row_size()?;
        }
    }
    Ok(())
}

fn validate_special_tokens(tokenizer: &Tokenizer) -> Result<()> {
    for text in [IM_START, IM_END, VISION_START, VISION_END] {
        let token = tokenizer
            .token_id(text)
            .with_context(|| format!("tokenizer is missing special token `{text}`"))?;
        ensure!(
            tokenizer.is_special(token),
            "token `{text}` ({token}) is not marked special"
        );
    }
    let im_end = tokenizer
        .token_id(IM_END)
        .with_context(|| format!("tokenizer is missing special token `{IM_END}`"))?;
    ensure!(
        tokenizer.eos_token() == im_end,
        "tokenizer EOS {} is not `{IM_END}` token {im_end}",
        tokenizer.eos_token()
    );
    Ok(())
}

#[derive(Default)]
enum JsonState {
    #[default]
    NotStarted,
    Normal,
    String {
        escaped: bool,
    },
    Invalid,
}

#[derive(Default)]
struct JsonObjectTracker {
    stack: Vec<char>,
    state: JsonState,
}

impl JsonObjectTracker {
    fn push(&mut self, text: &str) -> Option<usize> {
        for (offset, character) in text.char_indices() {
            let state = std::mem::take(&mut self.state);
            self.state = match state {
                JsonState::NotStarted => {
                    if character == '{' {
                        self.stack.push(character);
                        JsonState::Normal
                    } else {
                        JsonState::NotStarted
                    }
                }
                JsonState::Invalid => JsonState::Invalid,
                JsonState::Normal => match character {
                    '"' => JsonState::String { escaped: false },
                    '{' | '[' => {
                        self.stack.push(character);
                        JsonState::Normal
                    }
                    '}' => {
                        if self.stack.pop() != Some('{') {
                            JsonState::Invalid
                        } else if self.stack.is_empty() {
                            return Some(offset + character.len_utf8());
                        } else {
                            JsonState::Normal
                        }
                    }
                    ']' => {
                        if self.stack.pop() == Some('[') {
                            JsonState::Normal
                        } else {
                            JsonState::Invalid
                        }
                    }
                    _ => JsonState::Normal,
                },
                JsonState::String { escaped } => {
                    if escaped {
                        JsonState::String { escaped: false }
                    } else if character == '\\' {
                        JsonState::String { escaped: true }
                    } else if character == '"' {
                        JsonState::Normal
                    } else {
                        JsonState::String { escaped: false }
                    }
                }
            };
        }
        None
    }
}
