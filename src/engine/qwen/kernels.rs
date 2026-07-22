// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (c) 2026 Jarkko Sakkinen

//! Scalar CPU kernels used by the fixed Qwen inference path.
//!
//! `Q4_K`/`Q6_K` block decoding is derived from Dwarf Seek 4.

use std::sync::OnceLock;

use anyhow::{Result, bail, ensure};
use num_traits::ToPrimitive;
use rayon::prelude::*;

use super::gguf::{Tensor, TensorType};

/// Lossless `usize -> f32` for model dimensions/indexes that fit in `u16`.
/// `f32::from(u16)` is exact (24-bit mantissa holds any u16); `try_from` is the
/// clippy-recommended narrowing. Use only when the value provably fits `u16`.
pub(crate) fn dim_to_f32(value: usize) -> f32 {
    f32::from(u16::try_from(value).expect("model dimension/index exceeds u16"))
}

const Q8_0_BLOCK_VALUES: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 2 + Q8_0_BLOCK_VALUES;
const K_BLOCK_VALUES: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q6_K_BLOCK_BYTES: usize = 210;
const Q8_K_SUM_VALUES: usize = 16;
const PARALLEL_MIN_VALUES: usize = 16 * 1_024;
const MATRIX_ROW_TILE: usize = 4;
const MATRIX_OUTPUT_TILE: usize = 16;

/// Convert an IEEE 754 binary16 bit pattern to `f32`.
pub(crate) fn fp16_to_f32(value: u16) -> f32 {
    let sign = u32::from(value & 0x8000) << 16;
    let exponent = (value >> 10) & 0x1f;
    let fraction = value & 0x03ff;

    match exponent {
        0 if fraction == 0 => f32::from_bits(sign),
        0 => {
            let sign_multiplier = if sign == 0 { 1.0 } else { -1.0 };
            sign_multiplier * f32::from(fraction) * 2.0_f32.powi(-24)
        }
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (u32::from(fraction) << 13)),
        _ => f32::from_bits(
            sign | ((u32::from(exponent) + (127 - 15)) << 23) | (u32::from(fraction) << 13),
        ),
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Fp16AttentionKernel {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    AvxF16c,
}

impl Fp16AttentionKernel {
    pub(crate) fn detect() -> Self {
        static KERNEL: OnceLock<Fp16AttentionKernel> = OnceLock::new();

        *KERNEL.get_or_init(|| {
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("f16c") {
                return Self::AvxF16c;
            }
            Self::Scalar
        })
    }

    pub(crate) fn dot_pair(self, values: &[u16], left: &[f32], right: &[f32]) -> (f32, f32) {
        assert_eq!(values.len(), left.len(), "FP16 dot product lengths differ");
        assert_eq!(values.len(), right.len(), "FP16 dot product lengths differ");
        match self {
            Self::Scalar => fp16_dot_pair_scalar(values, left, right),
            #[cfg(target_arch = "x86_64")]
            Self::AvxF16c => {
                // The variant is only constructed after runtime feature detection.
                unsafe { fp16_dot_pair_avx_f16c(values, left, right) }
            }
        }
    }

    pub(crate) fn accumulate_pair(
        self,
        values: &[u16],
        left_weight: f32,
        right_weight: f32,
        left: &mut [f32],
        right: &mut [f32],
    ) {
        assert_eq!(values.len(), left.len(), "FP16 accumulation lengths differ");
        assert_eq!(
            values.len(),
            right.len(),
            "FP16 accumulation lengths differ"
        );
        match self {
            Self::Scalar => {
                fp16_accumulate_pair_scalar(values, left_weight, right_weight, left, right);
            }
            #[cfg(target_arch = "x86_64")]
            Self::AvxF16c => {
                // The variant is only constructed after runtime feature detection.
                unsafe {
                    fp16_accumulate_pair_avx_f16c(values, left_weight, right_weight, left, right);
                }
            }
        }
    }
}

fn fp16_dot_pair_scalar(values: &[u16], left: &[f32], right: &[f32]) -> (f32, f32) {
    let mut left_sum = 0.0;
    let mut right_sum = 0.0;
    for ((value, left), right) in values.iter().zip(left).zip(right) {
        let value = fp16_to_f32(*value);
        left_sum += left * value;
        right_sum += right * value;
    }
    (left_sum, right_sum)
}

fn fp16_accumulate_pair_scalar(
    values: &[u16],
    left_weight: f32,
    right_weight: f32,
    left: &mut [f32],
    right: &mut [f32],
) {
    for ((value, left), right) in values.iter().zip(left).zip(right) {
        let value = fp16_to_f32(*value);
        *left += left_weight * value;
        *right += right_weight * value;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn fp16_dot_pair_avx_f16c(values: &[u16], left: &[f32], right: &[f32]) -> (f32, f32) {
    use std::arch::x86_64::{
        __m128i, _mm256_add_ps, _mm256_cvtph_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps,
        _mm256_storeu_ps,
    };

    let vectorized_len = values.len() / 8 * 8;
    let mut left_sums = _mm256_setzero_ps();
    let mut right_sums = _mm256_setzero_ps();
    for index in (0..vectorized_len).step_by(8) {
        // `read_unaligned` is the intended unaligned read; codegen is identical
        // to `_mm_loadu_si128` and does not trigger `cast_ptr_alignment`.
        let packed =
            unsafe { std::ptr::read_unaligned(values.as_ptr().add(index).cast::<__m128i>()) };
        let unpacked = _mm256_cvtph_ps(packed);
        let left_values = unsafe { _mm256_loadu_ps(left.as_ptr().add(index)) };
        let right_values = unsafe { _mm256_loadu_ps(right.as_ptr().add(index)) };
        left_sums = _mm256_add_ps(left_sums, _mm256_mul_ps(left_values, unpacked));
        right_sums = _mm256_add_ps(right_sums, _mm256_mul_ps(right_values, unpacked));
    }
    let mut left_lanes = [0.0_f32; 8];
    let mut right_lanes = [0.0_f32; 8];
    unsafe {
        _mm256_storeu_ps(left_lanes.as_mut_ptr(), left_sums);
        _mm256_storeu_ps(right_lanes.as_mut_ptr(), right_sums);
    }
    let (left_tail, right_tail) = fp16_dot_pair_scalar(
        &values[vectorized_len..],
        &left[vectorized_len..],
        &right[vectorized_len..],
    );
    (
        left_lanes.into_iter().sum::<f32>() + left_tail,
        right_lanes.into_iter().sum::<f32>() + right_tail,
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn fp16_accumulate_pair_avx_f16c(
    values: &[u16],
    left_weight: f32,
    right_weight: f32,
    left: &mut [f32],
    right: &mut [f32],
) {
    use std::arch::x86_64::{
        __m128i, _mm256_add_ps, _mm256_cvtph_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps,
        _mm256_storeu_ps,
    };

    let vectorized_len = values.len() / 8 * 8;
    let left_weights = _mm256_set1_ps(left_weight);
    let right_weights = _mm256_set1_ps(right_weight);
    for index in (0..vectorized_len).step_by(8) {
        // `read_unaligned` is the intended unaligned read; codegen is identical
        // to `_mm_loadu_si128` and does not trigger `cast_ptr_alignment`.
        let packed =
            unsafe { std::ptr::read_unaligned(values.as_ptr().add(index).cast::<__m128i>()) };
        let unpacked = _mm256_cvtph_ps(packed);
        let left_values = unsafe { _mm256_loadu_ps(left.as_ptr().add(index)) };
        let right_values = unsafe { _mm256_loadu_ps(right.as_ptr().add(index)) };
        let left_values = _mm256_add_ps(left_values, _mm256_mul_ps(left_weights, unpacked));
        let right_values = _mm256_add_ps(right_values, _mm256_mul_ps(right_weights, unpacked));
        unsafe {
            _mm256_storeu_ps(left.as_mut_ptr().add(index), left_values);
            _mm256_storeu_ps(right.as_mut_ptr().add(index), right_values);
        }
    }
    fp16_accumulate_pair_scalar(
        &values[vectorized_len..],
        left_weight,
        right_weight,
        &mut left[vectorized_len..],
        &mut right[vectorized_len..],
    );
}

/// Convert `f32` to an IEEE 754 binary16 bit pattern using ties-to-even rounding.
pub(crate) fn f32_to_fp16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = u16::try_from(bits >> 16).expect("top 16 bits fit u16") & 0x8000;
    let exponent = i32::from(u8::try_from((bits >> 23) & 0xff).expect("exponent fits u8"));
    let fraction = bits & 0x007f_ffff;

    if exponent == 0xff {
        if fraction == 0 {
            return sign | 0x7c00;
        }
        return sign
            | 0x7e00
            | (u16::try_from(fraction >> 13).expect("fraction fits u16") & 0x01ff);
    }

    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let significand = fraction | 0x0080_0000;
        let shift = u32::try_from(14 - half_exponent).expect("shift is non-negative");
        let mut rounded = significand >> shift;
        let remainder = significand & ((1_u32 << shift) - 1);
        let halfway = 1_u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && rounded & 1 != 0) {
            rounded += 1;
        }
        return sign | u16::try_from(rounded).expect("rounded fits u16");
    }

    let mut rounded_fraction = fraction >> 13;
    let remainder = fraction & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && rounded_fraction & 1 != 0) {
        rounded_fraction += 1;
    }
    let mut encoded_exponent = u16::try_from(half_exponent).expect("half_exponent in 1..=0x1e");
    if rounded_fraction == 0x400 {
        rounded_fraction = 0;
        encoded_exponent += 1;
        if encoded_exponent == 0x1f {
            return sign | 0x7c00;
        }
    }
    sign | (encoded_exponent << 10)
        | u16::try_from(rounded_fraction).expect("rounded_fraction fits u16")
}

/// Dequantize one `Q4_K` row into a caller-provided buffer.
pub(crate) fn dequantize_q4_k_row(row: &[u8], output: &mut [f32]) -> Result<()> {
    validate_k_row(row, output.len(), Q4_K_BLOCK_BYTES, "Q4_K")?;
    for (block, values) in row
        .chunks_exact(Q4_K_BLOCK_BYTES)
        .zip(output.chunks_exact_mut(K_BLOCK_VALUES))
    {
        let scale = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let minimum = fp16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let quantized = &block[16..];
        for group in 0..8 {
            let (group_scale, group_minimum) = q4_k_scale_min(scales, group);
            let byte_offset = group / 2 * 32;
            let shift = group % 2 * 4;
            for index in 0..32 {
                let value = (quantized[byte_offset + index] >> shift) & 0x0f;
                values[group * 32 + index] = scale * f32::from(group_scale) * f32::from(value)
                    - minimum * f32::from(group_minimum);
            }
        }
    }
    Ok(())
}

/// Dequantize one `Q6_K` row into a caller-provided buffer.
pub(crate) fn dequantize_q6_k_row(row: &[u8], output: &mut [f32]) -> Result<()> {
    validate_k_row(row, output.len(), Q6_K_BLOCK_BYTES, "Q6_K")?;
    for (block, values) in row
        .chunks_exact(Q6_K_BLOCK_BYTES)
        .zip(output.chunks_exact_mut(K_BLOCK_VALUES))
    {
        let low = &block[..128];
        let high = &block[128..192];
        let scales = &block[192..208];
        let scale = fp16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        for half in 0..2 {
            let low = &low[half * 64..];
            let high = &high[half * 32..];
            let scales = &scales[half * 8..];
            let output_offset = half * 128;
            for index in 0..32 {
                let scale_index = index / 16;
                let q1 = i16::from(low[index] & 0x0f) | (i16::from(high[index] & 3) << 4);
                let q2 =
                    i16::from(low[index + 32] & 0x0f) | (i16::from((high[index] >> 2) & 3) << 4);
                let q3 = i16::from(low[index] >> 4) | (i16::from((high[index] >> 4) & 3) << 4);
                let q4 = i16::from(low[index + 32] >> 4) | (i16::from((high[index] >> 6) & 3) << 4);
                for (group, quantized) in [q1, q2, q3, q4].into_iter().enumerate() {
                    let group_scale = i8::from_ne_bytes([scales[group * 2 + scale_index]]);
                    values[output_offset + group * 32 + index] =
                        scale * f32::from(group_scale) * f32::from(quantized - 32);
                }
            }
        }
    }
    Ok(())
}

fn q4_k_scale_min(scales: &[u8], group: usize) -> (u8, u8) {
    if group < 4 {
        (scales[group] & 0x3f, scales[group + 4] & 0x3f)
    } else {
        (
            (scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4),
            (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
        )
    }
}

fn validate_k_row(row: &[u8], value_count: usize, block_bytes: usize, kind: &str) -> Result<()> {
    ensure!(
        value_count.is_multiple_of(K_BLOCK_VALUES),
        "{kind} row length {value_count} is not a multiple of {K_BLOCK_VALUES}"
    );
    let expected = value_count / K_BLOCK_VALUES * block_bytes;
    ensure!(
        row.len() == expected,
        "{kind} row has {} bytes, expected {expected} for {value_count} values",
        row.len()
    );
    Ok(())
}

struct Q8KActivation {
    scales: Vec<f32>,
    values: Vec<i8>,
    sums: Vec<i16>,
}

impl Q8KActivation {
    fn new(vector: &[f32]) -> Result<Self> {
        ensure!(
            vector.len().is_multiple_of(K_BLOCK_VALUES),
            "Q8_K activation length {} is not a multiple of {K_BLOCK_VALUES}",
            vector.len()
        );
        ensure!(
            vector.iter().all(|value| value.is_finite()),
            "Q8_K activation contains a non-finite value"
        );
        let block_count = vector.len() / K_BLOCK_VALUES;
        let mut scales = Vec::with_capacity(block_count);
        let mut values = Vec::with_capacity(vector.len());
        let mut sums = Vec::with_capacity(block_count * (K_BLOCK_VALUES / Q8_K_SUM_VALUES));
        for block in vector.chunks_exact(K_BLOCK_VALUES) {
            let maximum = block.iter().copied().fold(0.0_f32, |maximum, value| {
                if value.abs() > maximum.abs() {
                    value
                } else {
                    maximum
                }
            });
            if maximum == 0.0 {
                scales.push(0.0);
                values.resize(values.len() + K_BLOCK_VALUES, 0);
                sums.resize(sums.len() + K_BLOCK_VALUES / Q8_K_SUM_VALUES, 0);
                continue;
            }
            scales.push(-maximum / 127.0);
            let start = values.len();
            values.extend(block.iter().map(|value| {
                (-127.0 * (value / maximum))
                    .round_ties_even()
                    .clamp(-128.0, 127.0)
                    .to_i8()
                    .unwrap_or(0)
            }));
            sums.extend(
                values[start..]
                    .chunks_exact(Q8_K_SUM_VALUES)
                    .map(|group| group.iter().map(|value| i16::from(*value)).sum::<i16>()),
            );
        }
        Ok(Self {
            scales,
            values,
            sums,
        })
    }
}

#[derive(Clone, Copy)]
enum KDotKernel {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
}

impl KDotKernel {
    fn detect() -> Self {
        static KERNEL: OnceLock<KDotKernel> = OnceLock::new();
        *KERNEL.get_or_init(|| {
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
            Self::Scalar
        })
    }

    fn dot(self, kind: TensorType, row: &[u8], activation: &Q8KActivation) -> f32 {
        match (self, kind) {
            (Self::Scalar, TensorType::Q4K) => dot_q4_k_q8_k_scalar(row, activation),
            (Self::Scalar, TensorType::Q6K) => dot_q6_k_q8_k_scalar(row, activation),
            #[cfg(target_arch = "x86_64")]
            (Self::Avx2, TensorType::Q4K) => unsafe { dot_q4_k_q8_k_avx2(row, activation) },
            #[cfg(target_arch = "x86_64")]
            (Self::Avx2, TensorType::Q6K) => unsafe { dot_q6_k_q8_k_avx2(row, activation) },
            (_, kind) => unreachable!("unsupported K-quantized tensor type {kind:?}"),
        }
    }
}

fn dot_q4_k_q8_k_scalar(row: &[u8], activation: &Q8KActivation) -> f32 {
    let mut sum = 0.0;
    for (block_index, block) in row.chunks_exact(Q4_K_BLOCK_BYTES).enumerate() {
        let scale =
            activation.scales[block_index] * fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let minimum =
            -activation.scales[block_index] * fp16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let quantized = &block[16..];
        let values = &activation.values[block_index * K_BLOCK_VALUES..][..K_BLOCK_VALUES];
        let sums = &activation.sums[block_index * 16..][..16];
        let mut product_sum = 0_i32;
        let mut minimum_sum = 0_i32;
        for group in 0..8 {
            let (group_scale, group_minimum) = q4_k_scale_min(scales, group);
            minimum_sum += i32::from(group_minimum)
                * (i32::from(sums[group * 2]) + i32::from(sums[group * 2 + 1]));
            let byte_offset = group / 2 * 32;
            let shift = group % 2 * 4;
            for index in 0..32 {
                let weight = (quantized[byte_offset + index] >> shift) & 0x0f;
                product_sum += i32::from(group_scale)
                    * i32::from(weight)
                    * i32::from(values[group * 32 + index]);
            }
        }
        sum += scale * product_sum.to_f32().unwrap_or(0.0)
            + minimum * minimum_sum.to_f32().unwrap_or(0.0);
    }
    sum
}

fn dot_q6_k_q8_k_scalar(row: &[u8], activation: &Q8KActivation) -> f32 {
    let mut sum = 0.0;
    for (block_index, block) in row.chunks_exact(Q6_K_BLOCK_BYTES).enumerate() {
        let low = &block[..128];
        let high = &block[128..192];
        let scales = &block[192..208];
        let scale = activation.scales[block_index]
            * fp16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let values = &activation.values[block_index * K_BLOCK_VALUES..][..K_BLOCK_VALUES];
        let mut product_sum = 0_i64;
        for half in 0..2 {
            let low = &low[half * 64..];
            let high = &high[half * 32..];
            let scales = &scales[half * 8..];
            let values = &values[half * 128..];
            for index in 0..32 {
                let scale_index = index / 16;
                let q1 = i16::from(low[index] & 0x0f) | (i16::from(high[index] & 3) << 4);
                let q2 =
                    i16::from(low[index + 32] & 0x0f) | (i16::from((high[index] >> 2) & 3) << 4);
                let q3 = i16::from(low[index] >> 4) | (i16::from((high[index] >> 4) & 3) << 4);
                let q4 = i16::from(low[index + 32] >> 4) | (i16::from((high[index] >> 6) & 3) << 4);
                for (group, quantized) in [q1, q2, q3, q4].into_iter().enumerate() {
                    let group_scale = i8::from_ne_bytes([scales[group * 2 + scale_index]]);
                    product_sum += i64::from(group_scale)
                        * i64::from(quantized - 32)
                        * i64::from(values[group * 32 + index]);
                }
            }
        }
        sum += scale * product_sum.to_f32().unwrap_or(0.0);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q4_k_q8_k_avx2(row: &[u8], activation: &Q8KActivation) -> f32 {
    use std::arch::x86_64::{
        _mm256_and_si256, _mm256_loadu_si256, _mm256_set1_epi8, _mm256_srli_epi16,
    };

    let mask = _mm256_set1_epi8(0x0f);
    let mut sum = 0.0;
    for (block_index, block) in row.chunks_exact(Q4_K_BLOCK_BYTES).enumerate() {
        let scale =
            activation.scales[block_index] * fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let minimum =
            -activation.scales[block_index] * fp16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let quantized = &block[16..];
        let values = &activation.values[block_index * K_BLOCK_VALUES..][..K_BLOCK_VALUES];
        let sums = &activation.sums[block_index * 16..][..16];
        let mut product_sum = 0_i32;
        let mut minimum_sum = 0_i32;
        for group in 0..8 {
            let (group_scale, group_minimum) = q4_k_scale_min(scales, group);
            minimum_sum += i32::from(group_minimum)
                * (i32::from(sums[group * 2]) + i32::from(sums[group * 2 + 1]));
            let packed =
                unsafe { _mm256_loadu_si256(quantized.as_ptr().add(group / 2 * 32).cast()) };
            let unpacked = if group % 2 == 0 {
                _mm256_and_si256(packed, mask)
            } else {
                _mm256_and_si256(_mm256_srli_epi16::<4>(packed), mask)
            };
            let activation_values =
                unsafe { _mm256_loadu_si256(values.as_ptr().add(group * 32).cast()) };
            product_sum +=
                i32::from(group_scale) * unsafe { dot_u8_i8_avx2(unpacked, activation_values) };
        }
        sum += scale * product_sum.to_f32().unwrap_or(0.0)
            + minimum * minimum_sum.to_f32().unwrap_or(0.0);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q6_k_q8_k_avx2(row: &[u8], activation: &Q8KActivation) -> f32 {
    use std::arch::x86_64::_mm256_loadu_si256;

    let mut sum = 0.0;
    for (block_index, block) in row.chunks_exact(Q6_K_BLOCK_BYTES).enumerate() {
        let low = &block[..128];
        let high = &block[128..192];
        let scales = &block[192..208];
        let scale = activation.scales[block_index]
            * fp16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let values = &activation.values[block_index * K_BLOCK_VALUES..][..K_BLOCK_VALUES];
        let mut product_sum = 0_i64;
        for half in 0..2 {
            let low = &low[half * 64..];
            let high = &high[half * 32..];
            let scales = &scales[half * 8..];
            let values = &values[half * 128..];
            for group in 0..4 {
                let mut unpacked = [0_i8; 32];
                for index in 0..32 {
                    let low_value = match group {
                        0 => low[index] & 0x0f,
                        1 => low[index + 32] & 0x0f,
                        2 => low[index] >> 4,
                        3 => low[index + 32] >> 4,
                        _ => unreachable!(),
                    };
                    unpacked[index] = i8::try_from(
                        i16::from(low_value) | (i16::from((high[index] >> (group * 2)) & 3) << 4),
                    )
                    .unwrap_or(0)
                        - 32;
                }
                let weights = unsafe { _mm256_loadu_si256(unpacked.as_ptr().cast()) };
                let activation_values =
                    unsafe { _mm256_loadu_si256(values.as_ptr().add(group * 32).cast()) };
                let (first, second) = unsafe { dot_i8_i8_halves_avx2(weights, activation_values) };
                product_sum += i64::from(i8::from_ne_bytes([scales[group * 2]])) * i64::from(first);
                product_sum +=
                    i64::from(i8::from_ne_bytes([scales[group * 2 + 1]])) * i64::from(second);
            }
        }
        sum += scale * product_sum.to_f32().unwrap_or(0.0);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_u8_i8_avx2(
    left: std::arch::x86_64::__m256i,
    right: std::arch::x86_64::__m256i,
) -> i32 {
    use std::arch::x86_64::{
        _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi16, _mm256_storeu_si256,
    };
    let pairs = _mm256_maddubs_epi16(left, right);
    let lanes = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
    let mut sums = [0_i32; 8];
    unsafe { _mm256_storeu_si256(sums.as_mut_ptr().cast(), lanes) };
    sums.into_iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_i8_halves_avx2(
    left: std::arch::x86_64::__m256i,
    right: std::arch::x86_64::__m256i,
) -> (i32, i32) {
    use std::arch::x86_64::{_mm256_abs_epi8, _mm256_sign_epi8};
    let absolute = _mm256_abs_epi8(left);
    let signed = _mm256_sign_epi8(right, left);
    let pairs = unsafe { dot_u8_i8_lanes_avx2(absolute, signed) };
    (pairs[..4].iter().sum(), pairs[4..].iter().sum())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_u8_i8_lanes_avx2(
    left: std::arch::x86_64::__m256i,
    right: std::arch::x86_64::__m256i,
) -> [i32; 8] {
    use std::arch::x86_64::{
        _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi16, _mm256_storeu_si256,
    };
    let pairs = _mm256_maddubs_epi16(left, right);
    let lanes = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
    let mut sums = [0_i32; 8];
    unsafe { _mm256_storeu_si256(sums.as_mut_ptr().cast(), lanes) };
    sums
}

struct Q8Activation {
    scales: Vec<f32>,
    values: Vec<i8>,
}

impl Q8Activation {
    // `round()` + `clamp(-127.0, 127.0)` keeps the value in `i8` range, so the
    // checked `to_i8()` is total here and rejects only a contract bug.
    fn new(vector: &[f32]) -> Result<Self> {
        ensure!(
            vector.len().is_multiple_of(Q8_0_BLOCK_VALUES),
            "Q8 activation length {} is not a multiple of {Q8_0_BLOCK_VALUES}",
            vector.len()
        );
        ensure!(
            vector.iter().all(|value| value.is_finite()),
            "Q8 activation contains a non-finite value"
        );
        let mut scales = Vec::with_capacity(vector.len() / Q8_0_BLOCK_VALUES);
        let mut values = Vec::with_capacity(vector.len());
        for block in vector.chunks_exact(Q8_0_BLOCK_VALUES) {
            let maximum = block.iter().copied().map(f32::abs).fold(0.0, f32::max);
            let scale = maximum / 127.0;
            let inverse = if scale == 0.0 { 0.0 } else { scale.recip() };
            scales.push(scale);
            values.extend(block.iter().map(|value| {
                (value * inverse)
                    .round()
                    .clamp(-127.0, 127.0)
                    .to_i8()
                    .unwrap_or(0)
            }));
        }
        Ok(Self { scales, values })
    }
}

#[derive(Clone, Copy)]
enum Q8DotKernel {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl Q8DotKernel {
    fn detect() -> Self {
        static KERNEL: OnceLock<Q8DotKernel> = OnceLock::new();

        *KERNEL.get_or_init(|| {
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
            #[cfg(target_arch = "aarch64")]
            {
                return Self::Neon;
            }
            #[allow(unreachable_code)]
            Self::Scalar
        })
    }

    fn dot(self, row: &[u8], activation: &Q8Activation) -> f32 {
        match self {
            Self::Scalar => dot_q8_0_quantized_scalar(row, activation),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => {
                // The variant is only constructed after runtime AVX2 detection.
                unsafe { dot_q8_0_quantized_avx2(row, activation) }
            }
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { dot_q8_0_quantized_neon(row, activation) },
        }
    }
}

fn dot_q8_0_quantized_scalar(row: &[u8], activation: &Q8Activation) -> f32 {
    let mut sum = 0.0;
    for ((block, values), activation_scale) in row
        .chunks_exact(Q8_0_BLOCK_BYTES)
        .zip(activation.values.chunks_exact(Q8_0_BLOCK_VALUES))
        .zip(&activation.scales)
    {
        let weight_scale = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        // i8 weights and i8 activations each fit exactly in f32; the 32-wide
        // product sum stays below f32's 24-bit exact range.
        let block_sum = block[2..]
            .iter()
            .zip(values)
            .map(|(weight, value)| f32::from(i8::from_ne_bytes([*weight])) * f32::from(*value))
            .sum::<f32>();
        sum += weight_scale * activation_scale * block_sum;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_q8_0_quantized_neon(row: &[u8], activation: &Q8Activation) -> f32 {
    use std::arch::aarch64::*;
    let mut total = 0.0_f32;
    for ((block, values), activation_scale) in row
        .chunks_exact(Q8_0_BLOCK_BYTES)
        .zip(activation.values.chunks_exact(Q8_0_BLOCK_VALUES))
        .zip(&activation.scales)
    {
        let weight_scale = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let mut acc = vdupq_n_s32(0);
        let weights_ptr = block[2..].as_ptr().cast::<i8>();
        let values_ptr = values.as_ptr();

        let w0 = vld1q_s8(weights_ptr);
        let v0 = vld1q_s8(values_ptr);
        let w1 = vld1q_s8(weights_ptr.add(16));
        let v1 = vld1q_s8(values_ptr.add(16));

        let p0 = vmull_s8(vget_low_s8(w0), vget_low_s8(v0));
        let p1 = vmull_high_s8(w0, v0);
        let p2 = vmull_s8(vget_low_s8(w1), vget_low_s8(v1));
        let p3 = vmull_high_s8(w1, v1);

        acc = vaddq_s32(acc, vpaddlq_s16(p0));
        acc = vaddq_s32(acc, vpaddlq_s16(p1));
        acc = vaddq_s32(acc, vpaddlq_s16(p2));
        acc = vaddq_s32(acc, vpaddlq_s16(p3));

        let block_sum = vaddvq_s32(acc) as f32;
        total += weight_scale * activation_scale * block_sum;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q8_0_quantized_avx2(row: &[u8], activation: &Q8Activation) -> f32 {
    use std::arch::x86_64::{
        __m256i, _mm_add_epi32, _mm_cvtepi32_ps, _mm_cvtss_f32, _mm_shuffle_epi32,
        _mm_unpackhi_epi64, _mm256_abs_epi8, _mm256_castsi256_si128, _mm256_extracti128_si256,
        _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi16, _mm256_sign_epi8,
    };

    let ones = _mm256_set1_epi16(1);
    let mut sum = 0.0;
    for ((block, values), activation_scale) in row
        .chunks_exact(Q8_0_BLOCK_BYTES)
        .zip(activation.values.chunks_exact(Q8_0_BLOCK_VALUES))
        .zip(&activation.scales)
    {
        // `read_unaligned` is the intended unaligned read; codegen is identical
        // to `_mm256_loadu_si256` and does not trigger `cast_ptr_alignment`.
        let weights = unsafe { std::ptr::read_unaligned(block[2..].as_ptr().cast::<__m256i>()) };
        let activations = unsafe { std::ptr::read_unaligned(values.as_ptr().cast::<__m256i>()) };
        let signed_activations = _mm256_sign_epi8(activations, weights);
        let weight_magnitudes = _mm256_abs_epi8(weights);
        let pairs = _mm256_maddubs_epi16(weight_magnitudes, signed_activations);
        let products = _mm256_madd_epi16(pairs, ones);
        let low = _mm256_castsi256_si128(products);
        let high = _mm256_extracti128_si256::<1>(products);
        let lanes = _mm_add_epi32(low, high);
        let pairs = _mm_add_epi32(lanes, _mm_unpackhi_epi64(lanes, lanes));
        let total = _mm_add_epi32(pairs, _mm_shuffle_epi32::<0x55>(pairs));
        let weight_scale = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        sum += weight_scale * activation_scale * _mm_cvtss_f32(_mm_cvtepi32_ps(total));
    }
    sum
}

/// Return the index of the largest output from a quantized matrix-vector product.
pub(crate) fn matrix_vector_argmax(matrix: &Tensor, vector: &[f32]) -> Result<usize> {
    #[derive(Clone, Copy)]
    struct Candidate {
        invalid: Option<usize>,
        index: usize,
        value: f32,
    }

    enum Activation {
        Q8(Q8Activation, Q8DotKernel),
        K(Q8KActivation, KDotKernel),
    }

    let [input_size, output_size] = matrix_dimensions(matrix)?;
    ensure!(
        output_size != 0,
        "cannot select from an empty matrix output"
    );
    ensure!(
        vector.len() == input_size,
        "matrix input is {input_size}, but vector length is {}",
        vector.len()
    );
    let row_bytes = match matrix.tensor_type() {
        TensorType::Q8_0 => q8_matrix_row_bytes(matrix, input_size, output_size)?,
        TensorType::Q4K | TensorType::Q6K => k_matrix_row_bytes(matrix, input_size, output_size)?,
        TensorType::F32 => bail!("fused matrix argmax requires a quantized tensor"),
    };
    let activation = match matrix.tensor_type() {
        TensorType::Q8_0 => Activation::Q8(Q8Activation::new(vector)?, Q8DotKernel::detect()),
        TensorType::Q4K | TensorType::Q6K => {
            Activation::K(Q8KActivation::new(vector)?, KDotKernel::detect())
        }
        TensorType::F32 => unreachable!(),
    };
    let best = (0..output_size)
        .into_par_iter()
        .map(|index| {
            let start = index * row_bytes;
            let row = &matrix.data()[start..start + row_bytes];
            let value = match &activation {
                Activation::Q8(activation, kernel) => kernel.dot(row, activation),
                Activation::K(activation, kernel) => {
                    kernel.dot(matrix.tensor_type(), row, activation)
                }
            };
            Candidate {
                invalid: (!value.is_finite()).then_some(index),
                index,
                value,
            }
        })
        .reduce(
            || Candidate {
                invalid: None,
                index: 0,
                value: f32::NEG_INFINITY,
            },
            |left, right| {
                let take_right = right.value.partial_cmp(&left.value).is_some_and(|order| {
                    order.is_gt() || (order.is_eq() && right.index < left.index)
                });
                let (index, value) = if take_right {
                    (right.index, right.value)
                } else {
                    (left.index, left.value)
                };
                Candidate {
                    invalid: match (left.invalid, right.invalid) {
                        (Some(left), Some(right)) => Some(left.min(right)),
                        (left, right) => left.or(right),
                    },
                    index,
                    value,
                }
            },
        );
    if let Some(index) = best.invalid {
        bail!("matrix output {index} is not finite");
    }
    Ok(best.index)
}

/// Multiply row-major `F32` vectors by a GGUF matrix with dimensions `[input, output]`.
///
/// `vectors` contains `row_count` consecutive vectors. The result uses the same
/// row-major layout with `output` values per row.
pub(crate) fn matrix_matrix(
    matrix: &Tensor,
    vectors: &[f32],
    row_count: usize,
) -> Result<Vec<f32>> {
    let [input_size, output_size] = matrix_dimensions(matrix)?;
    let expected = row_count
        .checked_mul(input_size)
        .ok_or_else(|| anyhow::anyhow!("matrix-matrix input size overflow"))?;
    ensure!(
        vectors.len() == expected,
        "matrix-matrix input has {} values, expected {expected} ({row_count} x {input_size})",
        vectors.len()
    );
    if is_k_quantized(matrix.tensor_type()) {
        let activations = k_matrix_activations(vectors, row_count, input_size)?;
        return k_matrix_matrix(matrix, output_size, &activations);
    }
    let output_len = row_count
        .checked_mul(output_size)
        .ok_or_else(|| anyhow::anyhow!("matrix-matrix output size overflow"))?;
    let mut output = vec![0.0; output_len];

    match matrix.tensor_type() {
        TensorType::F32 => {
            let data = f32_matrix_data(matrix, input_size, output_size)?;
            let kernel = F32DotKernel::detect();
            if row_count == 1 {
                f32_matrix_vector(&mut output, data, input_size, vectors, kernel);
            } else {
                output
                    .par_chunks_mut(MATRIX_ROW_TILE * output_size)
                    .zip(vectors.par_chunks(MATRIX_ROW_TILE * input_size))
                    .for_each(|(output_rows, input_rows)| {
                        for output_start in (0..output_size).step_by(MATRIX_OUTPUT_TILE) {
                            let output_end = (output_start + MATRIX_OUTPUT_TILE).min(output_size);
                            for output_channel in output_start..output_end {
                                let row_start = output_channel * input_size;
                                let weights = &data[row_start..row_start + input_size];
                                for (output_row, input_row) in output_rows
                                    .chunks_exact_mut(output_size)
                                    .zip(input_rows.chunks_exact(input_size))
                                {
                                    output_row[output_channel] = kernel.dot(weights, input_row);
                                }
                            }
                        }
                    });
            }
        }
        TensorType::Q8_0 => {
            let row_bytes = q8_matrix_row_bytes(matrix, input_size, output_size)?;
            let kernel = Q8DotKernel::detect();
            if row_count == 1 {
                let activation = Q8Activation::new(vectors)?;
                q8_matrix_vector(&mut output, matrix.data(), row_bytes, &activation, kernel);
            } else {
                output
                    .par_chunks_mut(MATRIX_ROW_TILE * output_size)
                    .zip(vectors.par_chunks(MATRIX_ROW_TILE * input_size))
                    .try_for_each(|(output_rows, input_rows)| -> Result<()> {
                        let activations = input_rows
                            .chunks_exact(input_size)
                            .map(Q8Activation::new)
                            .collect::<Result<Vec<_>>>()?;
                        for output_start in (0..output_size).step_by(MATRIX_OUTPUT_TILE) {
                            let output_end = (output_start + MATRIX_OUTPUT_TILE).min(output_size);
                            for output_channel in output_start..output_end {
                                let row_start = output_channel * row_bytes;
                                let weights = &matrix.data()[row_start..row_start + row_bytes];
                                for (output_row, activation) in
                                    output_rows.chunks_exact_mut(output_size).zip(&activations)
                                {
                                    output_row[output_channel] = kernel.dot(weights, activation);
                                }
                            }
                        }
                        Ok(())
                    })?;
            }
        }
        TensorType::Q4K | TensorType::Q6K => unreachable!(),
    }

    Ok(output)
}

/// Multiply two matrices by shared row-major activation vectors.
pub(crate) fn matrix_matrix_pair(
    left: &Tensor,
    right: &Tensor,
    vectors: &[f32],
    row_count: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let dimensions = matrix_dimensions(left)?;
    ensure!(
        matrix_dimensions(right)? == dimensions,
        "paired matrix dimensions differ"
    );
    let [input_size, output_size] = dimensions;
    if left.tensor_type() == TensorType::Q8_0 && right.tensor_type() == TensorType::Q8_0 {
        let activations = matrix_activations(vectors, row_count, input_size)?;
        let left_row_bytes = q8_matrix_row_bytes(left, input_size, output_size)?;
        let right_row_bytes = q8_matrix_row_bytes(right, input_size, output_size)?;
        let outputs = rayon::join(
            || q8_matrix_matrix(left.data(), left_row_bytes, output_size, &activations),
            || q8_matrix_matrix(right.data(), right_row_bytes, output_size, &activations),
        );
        return Ok(outputs);
    }
    ensure!(
        is_k_quantized(left.tensor_type()) && is_k_quantized(right.tensor_type()),
        "paired matrix projection requires matching quantization families"
    );
    let activations = k_matrix_activations(vectors, row_count, input_size)?;
    let (left_output, right_output) = rayon::join(
        || k_matrix_matrix(left, output_size, &activations),
        || k_matrix_matrix(right, output_size, &activations),
    );
    Ok((left_output?, right_output?))
}

/// Multiply three matrices by shared row-major activation vectors.
pub(crate) fn matrix_matrix_triple(
    first: &Tensor,
    second: &Tensor,
    third: &Tensor,
    vectors: &[f32],
    row_count: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let [input_size, first_size] = matrix_dimensions(first)?;
    let [second_input, second_size] = matrix_dimensions(second)?;
    let [third_input, third_size] = matrix_dimensions(third)?;
    ensure!(
        second_input == input_size && third_input == input_size,
        "triple matrix input dimensions differ"
    );
    if first.tensor_type() == TensorType::Q8_0
        && second.tensor_type() == TensorType::Q8_0
        && third.tensor_type() == TensorType::Q8_0
    {
        let activations = matrix_activations(vectors, row_count, input_size)?;
        let first_row_bytes = q8_matrix_row_bytes(first, input_size, first_size)?;
        let second_row_bytes = q8_matrix_row_bytes(second, input_size, second_size)?;
        let third_row_bytes = q8_matrix_row_bytes(third, input_size, third_size)?;
        let (first_output, (second_output, third_output)) = rayon::join(
            || q8_matrix_matrix(first.data(), first_row_bytes, first_size, &activations),
            || {
                rayon::join(
                    || q8_matrix_matrix(second.data(), second_row_bytes, second_size, &activations),
                    || q8_matrix_matrix(third.data(), third_row_bytes, third_size, &activations),
                )
            },
        );
        return Ok((first_output, second_output, third_output));
    }
    ensure!(
        is_k_quantized(first.tensor_type())
            && is_k_quantized(second.tensor_type())
            && is_k_quantized(third.tensor_type()),
        "triple matrix projection requires matching quantization families"
    );
    let activations = k_matrix_activations(vectors, row_count, input_size)?;
    let (first_output, (second_output, third_output)) = rayon::join(
        || k_matrix_matrix(first, first_size, &activations),
        || {
            rayon::join(
                || k_matrix_matrix(second, second_size, &activations),
                || k_matrix_matrix(third, third_size, &activations),
            )
        },
    );
    Ok((first_output?, second_output?, third_output?))
}

fn matrix_activations(
    vectors: &[f32],
    row_count: usize,
    input_size: usize,
) -> Result<Vec<Q8Activation>> {
    let expected = row_count
        .checked_mul(input_size)
        .ok_or_else(|| anyhow::anyhow!("matrix-matrix input size overflow"))?;
    ensure!(
        vectors.len() == expected,
        "matrix-matrix input has {} values, expected {expected} ({row_count} x {input_size})",
        vectors.len()
    );
    vectors
        .chunks_exact(input_size)
        .map(Q8Activation::new)
        .collect()
}

fn k_matrix_activations(
    vectors: &[f32],
    row_count: usize,
    input_size: usize,
) -> Result<Vec<Q8KActivation>> {
    let expected = row_count
        .checked_mul(input_size)
        .ok_or_else(|| anyhow::anyhow!("matrix-matrix input size overflow"))?;
    ensure!(
        vectors.len() == expected,
        "matrix-matrix input has {} values, expected {expected} ({row_count} x {input_size})",
        vectors.len()
    );
    vectors
        .chunks_exact(input_size)
        .map(Q8KActivation::new)
        .collect()
}

fn k_matrix_matrix(
    matrix: &Tensor,
    output_size: usize,
    activations: &[Q8KActivation],
) -> Result<Vec<f32>> {
    let input_size = matrix.dimensions()[0];
    let row_bytes = k_matrix_row_bytes(matrix, input_size, output_size)?;
    let kernel = KDotKernel::detect();
    let kind = matrix.tensor_type();
    let output_len = activations
        .len()
        .checked_mul(output_size)
        .ok_or_else(|| anyhow::anyhow!("K-quantized matrix output size overflow"))?;
    let mut output = vec![0.0; output_len];
    output
        .par_chunks_mut(MATRIX_ROW_TILE * output_size)
        .zip(activations.par_chunks(MATRIX_ROW_TILE))
        .for_each(|(output_rows, activations)| {
            for output_start in (0..output_size).step_by(MATRIX_OUTPUT_TILE) {
                let output_end = (output_start + MATRIX_OUTPUT_TILE).min(output_size);
                for output_channel in output_start..output_end {
                    let row_start = output_channel * row_bytes;
                    let weights = &matrix.data()[row_start..row_start + row_bytes];
                    for (output_row, activation) in
                        output_rows.chunks_exact_mut(output_size).zip(activations)
                    {
                        output_row[output_channel] = kernel.dot(kind, weights, activation);
                    }
                }
            }
        });
    Ok(output)
}

fn is_k_quantized(kind: TensorType) -> bool {
    matches!(kind, TensorType::Q4K | TensorType::Q6K)
}

fn q8_matrix_matrix(
    matrix: &[u8],
    row_bytes: usize,
    output_size: usize,
    activations: &[Q8Activation],
) -> Vec<f32> {
    let mut output = vec![0.0; activations.len() * output_size];
    let kernel = Q8DotKernel::detect();
    if let [activation] = activations {
        q8_matrix_vector(&mut output, matrix, row_bytes, activation, kernel);
        return output;
    }
    output
        .par_chunks_mut(MATRIX_ROW_TILE * output_size)
        .zip(activations.par_chunks(MATRIX_ROW_TILE))
        .for_each(|(output_rows, activations)| {
            for output_start in (0..output_size).step_by(MATRIX_OUTPUT_TILE) {
                let output_end = (output_start + MATRIX_OUTPUT_TILE).min(output_size);
                for output_channel in output_start..output_end {
                    let row_start = output_channel * row_bytes;
                    let weights = &matrix[row_start..row_start + row_bytes];
                    for (output_row, activation) in
                        output_rows.chunks_exact_mut(output_size).zip(activations)
                    {
                        output_row[output_channel] = kernel.dot(weights, activation);
                    }
                }
            }
        });
    output
}

fn f32_matrix_vector(
    output: &mut [f32],
    matrix: &[f32],
    row_size: usize,
    vector: &[f32],
    kernel: F32DotKernel,
) {
    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(output_channel, value)| {
            let row_start = output_channel * row_size;
            *value = kernel.dot(&matrix[row_start..row_start + row_size], vector);
        });
}

fn q8_matrix_vector(
    output: &mut [f32],
    matrix: &[u8],
    row_bytes: usize,
    activation: &Q8Activation,
    kernel: Q8DotKernel,
) {
    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(output_channel, value)| {
            let row_start = output_channel * row_bytes;
            *value = kernel.dot(&matrix[row_start..row_start + row_bytes], activation);
        });
}

fn matrix_dimensions(matrix: &Tensor) -> Result<[usize; 2]> {
    let dimensions = match matrix.dimensions() {
        [input_size, output_size] => [*input_size, *output_size],
        dimensions => {
            bail!("matrix must have two logical dimensions [input, output], got {dimensions:?}")
        }
    };
    ensure!(
        dimensions[0] != 0 && dimensions[1] != 0,
        "matrix dimensions must be nonzero, got {dimensions:?}"
    );
    Ok(dimensions)
}

fn f32_matrix_data<'a>(
    matrix: &'a Tensor<'_>,
    input_size: usize,
    output_size: usize,
) -> Result<&'a [f32]> {
    let data = matrix.f32_slice()?;
    let expected = input_size
        .checked_mul(output_size)
        .ok_or_else(|| anyhow::anyhow!("F32 matrix element count overflow"))?;
    ensure!(
        data.len() == expected,
        "F32 matrix has {} values, expected {expected}",
        data.len()
    );
    Ok(data)
}

fn q8_matrix_row_bytes(matrix: &Tensor, input_size: usize, output_size: usize) -> Result<usize> {
    ensure!(
        input_size.is_multiple_of(Q8_0_BLOCK_VALUES),
        "Q8_0 matrix input dimension {input_size} is not a multiple of {Q8_0_BLOCK_VALUES}"
    );
    let row_bytes = input_size / Q8_0_BLOCK_VALUES * Q8_0_BLOCK_BYTES;
    let expected = row_bytes
        .checked_mul(output_size)
        .ok_or_else(|| anyhow::anyhow!("Q8_0 matrix byte size overflow"))?;
    ensure!(
        matrix.data().len() == expected,
        "Q8_0 matrix has {} data bytes, expected {expected}",
        matrix.data().len()
    );
    Ok(row_bytes)
}

fn k_matrix_row_bytes(matrix: &Tensor, input_size: usize, output_size: usize) -> Result<usize> {
    ensure!(
        input_size.is_multiple_of(K_BLOCK_VALUES),
        "K-quantized matrix input dimension {input_size} is not a multiple of {K_BLOCK_VALUES}"
    );
    let block_bytes = match matrix.tensor_type() {
        TensorType::Q4K => Q4_K_BLOCK_BYTES,
        TensorType::Q6K => Q6_K_BLOCK_BYTES,
        kind => bail!("tensor is {kind:?}, not K-quantized"),
    };
    let row_bytes = input_size / K_BLOCK_VALUES * block_bytes;
    let expected = row_bytes
        .checked_mul(output_size)
        .ok_or_else(|| anyhow::anyhow!("K-quantized matrix byte size overflow"))?;
    ensure!(
        matrix.data().len() == expected,
        "K-quantized matrix has {} data bytes, expected {expected}",
        matrix.data().len()
    );
    Ok(row_bytes)
}

#[derive(Clone, Copy)]
pub(crate) enum F32DotKernel {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2Fma,
}

impl F32DotKernel {
    pub(crate) fn detect() -> Self {
        static KERNEL: OnceLock<F32DotKernel> = OnceLock::new();

        *KERNEL.get_or_init(|| {
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return Self::Avx2Fma;
            }
            Self::Scalar
        })
    }

    pub(crate) fn dot(self, row: &[f32], vector: &[f32]) -> f32 {
        assert_eq!(row.len(), vector.len(), "F32 dot product lengths differ");
        match self {
            Self::Scalar => dot_f32_scalar(row, vector),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2Fma => {
                // The variant is only constructed after runtime AVX2 and FMA detection.
                unsafe { dot_f32_avx2_fma(row, vector) }
            }
        }
    }

    pub(crate) fn accumulate(self, output: &mut [f32], weight: f32, values: &[f32]) {
        assert_eq!(
            output.len(),
            values.len(),
            "F32 accumulation lengths differ"
        );
        match self {
            Self::Scalar => accumulate_f32_scalar(output, weight, values),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2Fma => {
                // The variant is only constructed after runtime AVX2 and FMA detection.
                unsafe { accumulate_f32_avx2_fma(output, weight, values) };
            }
        }
    }
}

fn dot_f32_scalar(row: &[f32], vector: &[f32]) -> f32 {
    row.iter()
        .zip(vector)
        .map(|(left, right)| left * right)
        .sum()
}

fn accumulate_f32_scalar(output: &mut [f32], weight: f32, values: &[f32]) {
    output
        .iter_mut()
        .zip(values)
        .for_each(|(output, value)| *output += weight * value);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2_fma(row: &[f32], vector: &[f32]) -> f32 {
    use std::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    let vectorized_len = row.len() / 8 * 8;
    let mut sums = _mm256_setzero_ps();
    for index in (0..vectorized_len).step_by(8) {
        let left = unsafe { _mm256_loadu_ps(row.as_ptr().add(index)) };
        let right = unsafe { _mm256_loadu_ps(vector.as_ptr().add(index)) };
        sums = _mm256_fmadd_ps(left, right, sums);
    }
    let mut lanes = [0.0_f32; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), sums) };
    lanes.into_iter().sum::<f32>()
        + dot_f32_scalar(&row[vectorized_len..], &vector[vectorized_len..])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn accumulate_f32_avx2_fma(output: &mut [f32], weight: f32, values: &[f32]) {
    use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};

    let vectorized_len = output.len() / 8 * 8;
    let weights = _mm256_set1_ps(weight);
    for index in (0..vectorized_len).step_by(8) {
        let output_values = unsafe { _mm256_loadu_ps(output.as_ptr().add(index)) };
        let input_values = unsafe { _mm256_loadu_ps(values.as_ptr().add(index)) };
        let accumulated = _mm256_fmadd_ps(weights, input_values, output_values);
        unsafe { _mm256_storeu_ps(output.as_mut_ptr().add(index), accumulated) };
    }
    accumulate_f32_scalar(
        &mut output[vectorized_len..],
        weight,
        &values[vectorized_len..],
    );
}

/// Add a channel bias to each row in place.
pub(crate) fn add_bias(values: &mut [f32], bias: &[f32]) -> Result<()> {
    ensure!(!bias.is_empty(), "bias must not be empty");
    ensure!(
        values.len().is_multiple_of(bias.len()),
        "value length {} is not divisible by bias length {}",
        values.len(),
        bias.len()
    );
    if values.len() < PARALLEL_MIN_VALUES {
        for row in values.chunks_mut(bias.len()) {
            row.iter_mut()
                .zip(bias)
                .for_each(|(value, bias)| *value += bias);
        }
    } else {
        values.par_chunks_mut(bias.len()).for_each(|row| {
            row.iter_mut()
                .zip(bias)
                .for_each(|(value, bias)| *value += bias);
        });
    }
    Ok(())
}

/// Apply RMS normalization independently to each row.
pub(crate) fn rms_norm(
    values: &[f32],
    width: usize,
    weight: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>> {
    validate_norm_shape(values, width, weight, None, epsilon)?;
    let mut output = vec![0.0; values.len()];
    let normalize = |output_row: &mut [f32], input_row: &[f32]| {
        let mean_square =
            input_row.iter().map(|value| value * value).sum::<f32>() / dim_to_f32(width);
        let scale = (mean_square + epsilon).sqrt().recip();
        for index in 0..width {
            output_row[index] = input_row[index] * scale * weight[index];
        }
    };
    if values.len() < PARALLEL_MIN_VALUES {
        output
            .chunks_mut(width)
            .zip(values.chunks(width))
            .for_each(|(output_row, input_row)| normalize(output_row, input_row));
    } else {
        output
            .par_chunks_mut(width)
            .zip(values.par_chunks(width))
            .for_each(|(output_row, input_row)| normalize(output_row, input_row));
    }
    Ok(output)
}

/// Apply `LayerNorm` independently to each row.
pub(crate) fn layer_norm(
    values: &[f32],
    width: usize,
    weight: &[f32],
    bias: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>> {
    validate_norm_shape(values, width, weight, Some(bias), epsilon)?;
    let mut output = vec![0.0; values.len()];
    let normalize = |output_row: &mut [f32], input_row: &[f32]| {
        let mean = input_row.iter().sum::<f32>() / dim_to_f32(width);
        let variance = input_row
            .iter()
            .map(|value| {
                let centered = value - mean;
                centered * centered
            })
            .sum::<f32>()
            / dim_to_f32(width);
        let scale = (variance + epsilon).sqrt().recip();
        for index in 0..width {
            output_row[index] = (input_row[index] - mean) * scale * weight[index] + bias[index];
        }
    };
    if values.len() < PARALLEL_MIN_VALUES {
        output
            .chunks_mut(width)
            .zip(values.chunks(width))
            .for_each(|(output_row, input_row)| normalize(output_row, input_row));
    } else {
        output
            .par_chunks_mut(width)
            .zip(values.par_chunks(width))
            .for_each(|(output_row, input_row)| normalize(output_row, input_row));
    }
    Ok(output)
}

fn validate_norm_shape(
    values: &[f32],
    width: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    epsilon: f32,
) -> Result<()> {
    ensure!(width != 0, "normalization width must not be zero");
    ensure!(
        values.len().is_multiple_of(width),
        "value length {} is not divisible by normalization width {width}",
        values.len()
    );
    ensure!(
        weight.len() == width,
        "normalization weight length is {}, expected {width}",
        weight.len()
    );
    if let Some(bias) = bias {
        ensure!(
            bias.len() == width,
            "normalization bias length is {}, expected {width}",
            bias.len()
        );
    }
    ensure!(
        epsilon.is_finite() && epsilon >= 0.0,
        "normalization epsilon must be finite and non-negative"
    );
    Ok(())
}

/// Apply `SiLU` in place.
pub(crate) fn silu(values: &mut [f32]) {
    let apply = |value: &mut f32| *value /= 1.0 + (-*value).exp();
    if values.len() < PARALLEL_MIN_VALUES {
        values.iter_mut().for_each(apply);
    } else {
        values.par_iter_mut().for_each(apply);
    }
}

/// Apply the GELU tanh approximation in place.
pub(crate) fn gelu(values: &mut [f32]) {
    const COEFFICIENT: f32 = 0.044_715;
    const SQRT_TWO_OVER_PI: f32 = 0.797_884_6;

    let apply = |value: &mut f32| {
        let input = *value;
        *value =
            0.5 * input * (1.0 + (SQRT_TWO_OVER_PI * (input + COEFFICIENT * input.powi(3))).tanh());
    };
    if values.len() < PARALLEL_MIN_VALUES {
        values.iter_mut().for_each(apply);
    } else {
        values.par_iter_mut().for_each(apply);
    }
}

/// Apply softmax in place. Empty slices are accepted.
pub(crate) fn softmax(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = values
        .iter_mut()
        .map(|value| {
            *value = (*value - maximum).exp();
            *value
        })
        .sum::<f32>();
    for value in values {
        *value /= sum;
    }
}

/// Add `right` to `left` in place.
pub(crate) fn vector_add(left: &mut [f32], right: &[f32]) -> Result<()> {
    ensure!(
        left.len() == right.len(),
        "vector lengths differ: {} and {}",
        left.len(),
        right.len()
    );
    if left.len() < PARALLEL_MIN_VALUES {
        left.iter_mut()
            .zip(right)
            .for_each(|(left, right)| *left += right);
    } else {
        left.par_iter_mut()
            .zip(right.par_iter())
            .for_each(|(left, right)| *left += right);
    }
    Ok(())
}

/// Multiply `left` by `right` in place.
pub(crate) fn vector_multiply(left: &mut [f32], right: &[f32]) -> Result<()> {
    ensure!(
        left.len() == right.len(),
        "vector lengths differ: {} and {}",
        left.len(),
        right.len()
    );
    if left.len() < PARALLEL_MIN_VALUES {
        left.iter_mut()
            .zip(right)
            .for_each(|(left, right)| *left *= right);
    } else {
        left.par_iter_mut()
            .zip(right.par_iter())
            .for_each(|(left, right)| *left *= right);
    }
    Ok(())
}
