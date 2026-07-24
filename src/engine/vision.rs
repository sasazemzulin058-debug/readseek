// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (c) 2026 Jarkko Sakkinen

//! `Qwen3-VL` image analysis through `ReadSeek`'s fixed CPU inference engine. One
//! multimodal model handles captioning, object detection, and OCR; its GGUF
//! files are fetched lazily into the user cache directory.

// Coordinate conversion from normalized model output to pixels is bounded by
// the image dimensions.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

use std::env;
use std::io::IsTerminal as _;
use std::str::FromStr;
use std::sync::mpsc::{RecvTimeoutError, Sender, channel};
use std::sync::{Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, anyhow, bail};
use indicatif::ProgressBar;
use rayon::{ThreadPool, ThreadPoolBuilder};
use serde::{Deserialize, Serialize};
use strum_macros::{AsRefStr, Display};

use crate::engine::qwen::{SpatialReduction, TextModel, VisionEmbedding, VisionInput, VisionModel};

const MODEL_FILE: &str = "Qwen3VL-2B-Instruct-Q4_K_M.gguf";
const MMPROJ_FILE: &str = "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf";
const CAPTION_MAX_NEW_TOKENS: usize = 512;
const OBJECTS_MAX_NEW_TOKENS: usize = 2048;
const OCR_MAX_NEW_TOKENS: usize = 4096;
const LOCATION_BINS: i32 = 1000;
const PROGRESS_DEADLINE: Duration = Duration::from_secs(2);
const PROGRESS_TICK: Duration = Duration::from_millis(100);
const PROGRESS_MSG: &str = "Analyzing image...";

const FIELD_CAPTION: &str =
    "\"caption\": one concise paragraph of at most 100 words describing the image";
const FIELD_OBJECTS: &str = "\"objects\": an array of at most 32 {\"label\": string, \"bbox\": [x1,y1,x2,y2]} entries for the salient objects, where bbox is an axis-aligned box with integer coordinates normalized to 0-1000 relative to image width and height";
const FIELD_OCR: &str =
    "\"ocr\": a string containing all visible text in reading order, preserving line breaks";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize, Display, AsRefStr)]
#[strum(serialize_all = "lowercase")]
pub(crate) enum VisionLevel {
    #[default]
    Low,
    Medium,
    High,
}

impl FromStr for VisionLevel {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            _ => Err(format!(
                "unknown vision level `{value}`; expected low, medium, or high"
            )),
        }
    }
}

impl VisionLevel {
    fn image_max_tokens(self, request: Request) -> usize {
        let (caption, objects, ocr) = match self {
            Self::Low => (256, 768, 1024),
            Self::Medium => (768, 1024, 1536),
            Self::High => return 2048,
        };
        [
            request.caption.then_some(caption),
            request.objects.then_some(objects),
            request.ocr.then_some(ocr),
        ]
        .into_iter()
        .flatten()
        .max()
        .unwrap_or(0)
    }

    fn spatial_reduction(self) -> SpatialReduction {
        match self {
            Self::Low => SpatialReduction::MergeHalf,
            Self::Medium => SpatialReduction::PruneQuarter,
            Self::High => SpatialReduction::None,
        }
    }
}

/// A detected object with its category label and bounding box `[x1,y1,x2,y2]`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct DetectedObject {
    label: String,
    bbox: [i32; 4],
}

/// Which vision tasks to run against an image.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Request {
    pub(crate) caption: bool,
    pub(crate) objects: bool,
    pub(crate) ocr: bool,
}

impl VisionInput<'_> {
    pub(crate) fn cache_hash(self) -> String {
        match self {
            Self::Encoded(bytes) => crate::engine::hash::hash_bytes(bytes),
            Self::Rgb {
                width,
                height,
                pixels,
            } => {
                let mut hasher = blake3::Hasher::new();
                hasher.update(b"readseek-rgb\0");
                hasher.update(&width.to_le_bytes());
                hasher.update(&height.to_le_bytes());
                hasher.update(pixels);
                hasher.finalize().to_string()
            }
        }
    }

    /// Image dimensions without decoding the pixel data.
    fn dimensions(&self) -> Result<(u32, u32)> {
        match self {
            Self::Encoded(bytes) => {
                let (width, height) = image::ImageReader::new(std::io::Cursor::new(bytes))
                    .with_guessed_format()
                    .context("guess image format for Qwen3-VL")?
                    .into_dimensions()
                    .context("read image dimensions for Qwen3-VL")?;
                Ok((width, height))
            }
            Self::Rgb { width, height, .. } => Ok((*width, *height)),
        }
    }
}

/// Results of the requested vision tasks.
#[derive(Default)]
pub(crate) struct Analysis {
    pub(crate) caption: Option<String>,
    pub(crate) objects: Option<Vec<DetectedObject>>,
    pub(crate) ocr: Option<String>,
}

struct VisionRuntime {
    text: TextModel,
    vision: VisionModel,
    threads: usize,
    unreported_startup_ms: Option<u128>,
}

impl VisionRuntime {
    fn load(threads: usize) -> Result<Self> {
        let started = Instant::now();
        let model_path = crate::engine::model::file(MODEL_FILE)?;
        let mmproj_path = crate::engine::model::file(MMPROJ_FILE)?;
        let text = TextModel::load(model_path).context("load Qwen3-VL text model")?;
        let vision = VisionModel::load(mmproj_path).context("load Qwen3-VL vision projector")?;
        Ok(Self {
            text,
            vision,
            threads,
            unreported_startup_ms: Some(started.elapsed().as_millis()),
        })
    }
}

static INFERENCE_POOL: OnceLock<std::result::Result<ThreadPool, String>> = OnceLock::new();
static RUNTIME: OnceLock<std::result::Result<Mutex<VisionRuntime>, String>> = OnceLock::new();

fn inference_pool() -> Result<&'static ThreadPool> {
    INFERENCE_POOL
        .get_or_init(|| build_inference_pool().map_err(|error| format!("{error:#}")))
        .as_ref()
        .map_err(|error| anyhow!(error.clone()))
}

fn with_runtime<T: Send>(run: impl FnOnce(&mut VisionRuntime) -> Result<T> + Send) -> Result<T> {
    let pool = inference_pool()?;
    pool.install(|| {
        let runtime = RUNTIME.get_or_init(|| {
            VisionRuntime::load(pool.current_num_threads())
                .map(Mutex::new)
                .map_err(|error| format!("{error:#}"))
        });
        let runtime = runtime.as_ref().map_err(|error| anyhow!(error.clone()))?;
        let mut runtime = runtime
            .lock()
            .map_err(|_| anyhow!("vision runtime mutex is poisoned"))?;
        run(&mut runtime)
    })
}

/// Run the selected tasks in one multimodal generation pass. The loaded model
/// is reused by later images, which matters for PDFs containing several images.
pub(crate) fn analyze(
    input: VisionInput<'_>,
    request: Request,
    level: VisionLevel,
) -> Result<Analysis> {
    if !request.caption && !request.objects && !request.ocr {
        return Ok(Analysis::default());
    }

    let embedded_ocr = match input {
        VisionInput::Encoded(bytes) if request.ocr => {
            crate::engine::image::embedded_drawio_text(bytes)
        }
        _ => None,
    };
    let model_request = Request {
        caption: request.caption,
        objects: request.objects,
        ocr: request.ocr && embedded_ocr.is_none(),
    };
    if !model_request.caption && !model_request.objects && !model_request.ocr {
        return Ok(Analysis {
            ocr: embedded_ocr,
            ..Analysis::default()
        });
    }

    let image_max_tokens = level.image_max_tokens(model_request);
    let (raw, width, height) = with_runtime(|runtime| {
        let total_started = Instant::now();
        let startup_ms = runtime.unreported_startup_ms.take().unwrap_or(0);
        let _progress = InferenceProgress::new();
        let vision_started = Instant::now();
        let (width, height) = input.dimensions()?;
        let embedding = runtime
            .vision
            .encode_input(input, image_max_tokens, level.spatial_reduction())
            .context("encode image for Qwen3-VL")?;
        let vision_encode_ms = vision_started.elapsed().as_millis();
        let (raw, generation) = generate(runtime, &embedding, model_request)?;
        let prefill_tokens_per_second =
            tokens_per_second(generation.prompt_tokens, generation.prefill_ms);
        let decode_tokens_per_second =
            tokens_per_second(generation.generated_tokens, generation.decode_ms);
        tracing::trace!(
            target: "tracing",
            vision_level = ?level,
            image_max_tokens,
            image_tokens_before_reduction = embedding.original_token_count,
            image_tokens_after_reduction = embedding.token_count,
            prompt_tokens = generation.prompt_tokens,
            generated_tokens = generation.generated_tokens,
            threads = runtime.threads,
            startup_ms,
            vision_encode_ms,
            tokenize_ms = generation.tokenize_ms,
            prefill_ms = generation.prefill_ms,
            decode_ms = generation.decode_ms,
            prefill_tokens_per_second = ?prefill_tokens_per_second,
            decode_tokens_per_second = ?decode_tokens_per_second,
            total_ms = total_started.elapsed().as_millis(),
            peak_rss_bytes = ?peak_rss_bytes(),
            "vision inference completed"
        );
        Ok((raw, width, height))
    })?;
    let mut analysis = parse_analysis(&raw, model_request, width, height)?;
    if embedded_ocr.is_some() {
        analysis.ocr = embedded_ocr;
    }
    Ok(analysis)
}

fn build_prompt(request: Request) -> String {
    let mut fields = Vec::new();
    if request.caption {
        fields.push(FIELD_CAPTION);
    }
    if request.objects {
        fields.push(FIELD_OBJECTS);
    }
    if request.ocr {
        fields.push(FIELD_OCR);
    }
    format!(
        "Analyze the image and respond with a single JSON object containing {}. Output only the JSON object.",
        fields.join(", ")
    )
}

fn generate(
    runtime: &VisionRuntime,
    image: &VisionEmbedding,
    request: Request,
) -> Result<(String, GenerationMetrics)> {
    let prompt = build_prompt(request);
    let token_limit = max_new_tokens(request);
    let generation_started = Instant::now();
    let generation = runtime
        .text
        .generate(&prompt, image, token_limit)
        .context("generate Qwen3-VL response")?;
    let generation_duration = generation_started.elapsed();
    let measured_duration = generation
        .prefill_duration
        .saturating_add(generation.decode_duration);
    let tokenize_ms = generation_duration
        .saturating_sub(measured_duration)
        .as_millis();
    let metrics = GenerationMetrics {
        prompt_tokens: generation.prompt_tokens,
        generated_tokens: generation.generated_tokens,
        tokenize_ms,
        prefill_ms: generation.prefill_duration.as_millis(),
        decode_ms: generation.decode_duration.as_millis(),
    };
    Ok((generation.text, metrics))
}

struct GenerationMetrics {
    prompt_tokens: usize,
    generated_tokens: usize,
    tokenize_ms: u128,
    prefill_ms: u128,
    decode_ms: u128,
}

fn max_new_tokens(request: Request) -> usize {
    [
        request.caption.then_some(CAPTION_MAX_NEW_TOKENS),
        request.objects.then_some(OBJECTS_MAX_NEW_TOKENS),
        request.ocr.then_some(OCR_MAX_NEW_TOKENS),
    ]
    .into_iter()
    .flatten()
    .max()
    .unwrap_or(0)
}

fn tokens_per_second(tokens: usize, milliseconds: u128) -> Option<f64> {
    (milliseconds > 0).then_some(tokens as f64 * 1000.0 / milliseconds as f64)
}

#[derive(serde::Deserialize)]
struct ObjectJson {
    label: String,
    bbox: Vec<i32>,
}

#[derive(serde::Deserialize)]
struct CombinedJson {
    caption: Option<String>,
    objects: Option<Vec<ObjectJson>>,
    ocr: Option<String>,
}

fn parse_analysis(raw: &str, request: Request, width: u32, height: u32) -> Result<Analysis> {
    let json = extract_json(raw).context("vision response did not contain JSON")?;
    let parsed = match serde_json::from_str::<CombinedJson>(json) {
        Ok(parsed) => parsed,
        Err(error) => {
            let Some(parsed) = recover_json(raw) else {
                return Err(error).context("parse vision JSON response");
            };
            tracing::warn!("vision response was truncated; returning completed JSON prefix");
            parsed
        }
    };

    Ok(Analysis {
        caption: request
            .caption
            .then(|| parsed.caption.map(|value| strip_special(&value)))
            .flatten(),
        objects: request
            .objects
            .then(|| {
                parsed
                    .objects
                    .map(|objects| build_objects(objects, width, height))
            })
            .flatten(),
        ocr: request
            .ocr
            .then(|| parsed.ocr.map(|value| strip_special(&value)))
            .flatten(),
    })
}

fn build_objects(objects: Vec<ObjectJson>, width: u32, height: u32) -> Vec<DetectedObject> {
    objects
        .into_iter()
        .filter_map(|object| {
            if object.label.is_empty() || object.bbox.len() != 4 {
                return None;
            }
            let bbox = [
                location_to_pixel(object.bbox[0], width),
                location_to_pixel(object.bbox[1], height),
                location_to_pixel(object.bbox[2], width),
                location_to_pixel(object.bbox[3], height),
            ];
            (bbox[0] < bbox[2] && bbox[1] < bbox[3]).then_some(DetectedObject {
                label: object.label,
                bbox,
            })
        })
        .collect()
}

fn location_to_pixel(location: i32, dimension: u32) -> i32 {
    let location = location.clamp(0, LOCATION_BINS);
    (f64::from(location) / f64::from(LOCATION_BINS) * f64::from(dimension)).round() as i32
}

fn extract_json(raw: &str) -> Option<&str> {
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    raw.get(start..=end)
}

/// Recover the longest valid JSON prefix from a truncated model response.
fn recover_json(raw: &str) -> Option<CombinedJson> {
    let json = raw.get(raw.find('{')?..)?.trim_end();
    if let Some(parsed) = parse_completed_json(json) {
        return Some(parsed);
    }
    json.char_indices()
        .rev()
        .filter(|(_, character)| *character == '}')
        .find_map(|(offset, _)| parse_completed_json(&json[..=offset]))
}

fn parse_completed_json(json: &str) -> Option<CombinedJson> {
    let json = complete_json(json)?;
    serde_json::from_str(&json).ok()
}

fn complete_json(json: &str) -> Option<String> {
    let mut stack = Vec::new();
    let mut in_string = false;
    let mut escaped = false;

    for character in json.chars() {
        if in_string {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
            continue;
        }
        match character {
            '"' => in_string = true,
            '{' | '[' => stack.push(character),
            '}' if stack.pop() == Some('{') => {}
            ']' if stack.pop() == Some('[') => {}
            '}' | ']' => return None,
            _ => {}
        }
    }
    if in_string || stack.is_empty() {
        return None;
    }

    let mut completed = json.trim_end_matches(',').trim_end().to_owned();
    for opening in stack.iter().rev() {
        completed.push(match opening {
            '{' => '}',
            '[' => ']',
            _ => return None,
        });
    }
    Some(completed)
}

fn strip_special(raw: &str) -> String {
    raw.replace("<|im_end|>", "").trim().to_owned()
}

fn build_inference_pool() -> Result<ThreadPool> {
    let threads = inference_threads()?;
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .context("create vision inference thread pool")
}

fn inference_threads() -> Result<usize> {
    let available = std::thread::available_parallelism()
        .context("detect available parallelism for vision inference")?
        .get();
    let Some(value) = env::var_os("READSEEK_VISION_THREADS") else {
        return Ok(available);
    };
    let value = value
        .into_string()
        .map_err(|_| anyhow!("READSEEK_VISION_THREADS is not valid UTF-8"))?;
    let threads = value
        .parse::<usize>()
        .context("parse READSEEK_VISION_THREADS as a positive integer")?;
    if threads == 0 {
        bail!("READSEEK_VISION_THREADS must be greater than zero");
    }
    Ok(threads)
}

#[cfg(target_os = "linux")]
fn peak_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kibibytes = status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))?
        .split_ascii_whitespace()
        .next()?
        .parse::<u64>()
        .ok()?;
    kibibytes.checked_mul(1024)
}

#[cfg(not(target_os = "linux"))]
fn peak_rss_bytes() -> Option<u64> {
    None
}

#[derive(Default)]
struct InferenceProgress {
    completion: Option<Sender<()>>,
    worker: Option<JoinHandle<()>>,
}

impl InferenceProgress {
    fn new() -> Self {
        if !std::io::stderr().is_terminal() {
            return Self::default();
        }

        let (completion, receiver) = channel();
        let worker = std::thread::spawn(move || {
            match receiver.recv_timeout(PROGRESS_DEADLINE) {
                Ok(()) | Err(RecvTimeoutError::Disconnected) => return,
                Err(RecvTimeoutError::Timeout) => {}
            }
            let bar = ProgressBar::new_spinner();
            bar.set_message(PROGRESS_MSG);
            bar.enable_steady_tick(PROGRESS_TICK);
            let _ = receiver.recv();
            bar.finish_and_clear();
        });
        Self {
            completion: Some(completion),
            worker: Some(worker),
        }
    }
}

impl Drop for InferenceProgress {
    fn drop(&mut self) {
        self.completion.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
