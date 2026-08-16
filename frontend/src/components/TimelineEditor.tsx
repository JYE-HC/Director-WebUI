import { useEffect, useMemo, useRef, useState } from "react";
import { directorApi } from "../api/client";
import type { RV2VShotDetectionRequest } from "../api/types";
import {
  createLocalId,
  createTimelineItem,
  type ModeDraft,
  type RV2VDraft,
  type ShotBase,
} from "../domain/modes";
import { effectiveDraftTiming, effectiveShotTiming } from "../domain/timing";
import type {
  AssetMutation,
  DraftAssetField,
} from "../state/directorState";
import { MINIMAX_H3_REFERENCE_LIMITS } from "../domain/h3Capabilities";
import { limitPromptCharacters } from "../domain/promptLimits";
import { AssetUploader } from "./AssetUploader";
import { DeferredNumberInput, Field, Panel } from "./ui";

interface TimelineEditorProps {
  draft: ModeDraft;
  runtimeEnabled: boolean;
  onChange: (draft: ModeDraft) => void;
  onAssetsChange: (
    mode: ModeDraft["mode"],
    shotId: string,
    field: DraftAssetField,
    mutation: AssetMutation,
  ) => void;
}

function replaceBaseShot(draft: ModeDraft, index: number, patch: Partial<ShotBase>): ModeDraft {
  switch (draft.mode) {
    case "t2v": return { ...draft, shots: draft.shots.map((shot, i) => i === index ? { ...shot, ...patch } : shot) };
    case "i2v": return { ...draft, shots: draft.shots.map((shot, i) => i === index ? { ...shot, ...patch } : shot) };
    case "fl2v": return { ...draft, shots: draft.shots.map((shot, i) => i === index ? { ...shot, ...patch } : shot) };
    case "r2v": return { ...draft, shots: draft.shots.map((shot, i) => i === index ? { ...shot, ...patch } : shot) };
    case "v2v": return { ...draft, shots: draft.shots.map((shot, i) => i === index ? { ...shot, ...patch } : shot) };
    case "rv2v": return { ...draft, shots: draft.shots.map((shot, i) => i === index ? { ...shot, ...patch } : shot) };
  }
}

function addShot(draft: ModeDraft): ModeDraft {
  const index = draft.shots.length + 1;
  switch (draft.mode) {
    case "t2v": return { ...draft, shots: [...draft.shots, createTimelineItem("t2v", index)] };
    case "i2v": return { ...draft, shots: [...draft.shots, createTimelineItem("i2v", index)] };
    case "fl2v": return { ...draft, shots: [...draft.shots, createTimelineItem("fl2v", index)] };
    case "r2v": return { ...draft, shots: [...draft.shots, createTimelineItem("r2v", index)] };
    case "v2v": return { ...draft, shots: [...draft.shots, createTimelineItem("v2v", index)] };
    case "rv2v": return { ...draft, shots: [...draft.shots, createTimelineItem("rv2v", index)] };
  }
}

function removeShot(draft: ModeDraft, index: number): ModeDraft {
  switch (draft.mode) {
    case "t2v": return { ...draft, shots: draft.shots.filter((_, i) => i !== index) };
    case "i2v": return { ...draft, shots: draft.shots.filter((_, i) => i !== index) };
    case "fl2v": return { ...draft, shots: draft.shots.filter((_, i) => i !== index) };
    case "r2v": return { ...draft, shots: draft.shots.filter((_, i) => i !== index) };
    case "v2v": return { ...draft, shots: draft.shots.filter((_, i) => i !== index) };
    case "rv2v": return { ...draft, shots: draft.shots.filter((_, i) => i !== index) };
  }
}

export function splitRV2VShot(
  draft: RV2VDraft,
  shotId: string,
  offsetSeconds: number,
): RV2VDraft {
  if (draft.shots.length >= 128) return draft;
  const index = draft.shots.findIndex((shot) => shot.id === shotId);
  if (index < 0) return draft;
  const shot = draft.shots[index];
  if (
    !shot.source_video ||
    !Number.isFinite(offsetSeconds) ||
    offsetSeconds <= 0.01 ||
    offsetSeconds >= shot.duration_seconds - 0.01
  ) return draft;
  const ratio = offsetSeconds / shot.duration_seconds;
  const sourceOffset = shot.source_duration_seconds * ratio;
  if (sourceOffset <= 0.01 || sourceOffset >= shot.source_duration_seconds - 0.01)
    return draft;
  const left = {
    ...shot,
    duration_seconds: offsetSeconds,
    source_duration_seconds: sourceOffset,
  };
  const right = {
    ...structuredClone(shot),
    id: createLocalId("rv2v-shot"),
    title: `${shot.title || `片段 ${index + 1}`} · 后半段`,
    duration_seconds: shot.duration_seconds - offsetSeconds,
    source_start_seconds: shot.source_start_seconds + sourceOffset,
    source_duration_seconds: shot.source_duration_seconds - sourceOffset,
  };
  return {
    ...draft,
    shots: [
      ...draft.shots.slice(0, index),
      left,
      right,
      ...draft.shots.slice(index + 1),
    ],
  };
}

export function moveRV2VShot(
  draft: RV2VDraft,
  shotId: string,
  direction: -1 | 1,
): RV2VDraft {
  const from = draft.shots.findIndex((shot) => shot.id === shotId);
  const to = from + direction;
  if (from < 0 || to < 0 || to >= draft.shots.length) return draft;
  const shots = [...draft.shots];
  [shots[from], shots[to]] = [shots[to], shots[from]];
  return { ...draft, shots };
}

export function deleteRV2VShot(
  draft: RV2VDraft,
  shotId: string,
): RV2VDraft {
  if (draft.shots.length <= 1 || !draft.shots.some((shot) => shot.id === shotId))
    return draft;
  return { ...draft, shots: draft.shots.filter((shot) => shot.id !== shotId) };
}

/**
 * Maps full-source scene cuts into the selected clip's output-time domain.
 * A clip can be retimed, so source seconds must be scaled by
 * output-duration/source-duration before the existing proportional splitter
 * is applied.
 */
export function splitRV2VShotAtSourceCuts(
  draft: RV2VDraft,
  shotId: string,
  cutFrames: number[],
  frameRate: number,
): RV2VDraft {
  const original = draft.shots.find((shot) => shot.id === shotId);
  if (
    !original?.source_video ||
    !Number.isFinite(frameRate) ||
    frameRate <= 0 ||
    original.source_duration_seconds <= 0 ||
    original.duration_seconds <= 0
  ) return draft;
  const sourceStart = original.source_start_seconds;
  const sourceEnd = sourceStart + original.source_duration_seconds;
  const outputOffsets = [...new Set(cutFrames)]
    .filter((frame) => Number.isFinite(frame) && frame >= 0)
    .map((frame) => frame / frameRate)
    .filter((sourceTime) => sourceTime > sourceStart + 0.001 && sourceTime < sourceEnd - 0.001)
    .map((sourceTime) =>
      (sourceTime - sourceStart) /
      original.source_duration_seconds *
      original.duration_seconds,
    )
    .sort((left, right) => left - right);
  if (!outputOffsets.length) return draft;

  let next = draft;
  let currentId = shotId;
  let consumedOutput = 0;
  for (const outputOffset of outputOffsets) {
    if (next.shots.length >= 128) break;
    const currentIndex = next.shots.findIndex((shot) => shot.id === currentId);
    if (currentIndex < 0) break;
    const split = splitRV2VShot(next, currentId, outputOffset - consumedOutput);
    if (split === next) continue;
    next = split;
    currentId = next.shots[currentIndex + 1].id;
    consumedOutput = outputOffset;
  }
  return next;
}

function RV2VTimelineTrack({
  draft,
  selectedId,
  runtimeEnabled,
  onSelected,
  onChange,
}: {
  draft: RV2VDraft;
  selectedId: string;
  runtimeEnabled: boolean;
  onSelected: (shotId: string) => void;
  onChange: (draft: RV2VDraft) => void;
}) {
  const ranges = useMemo(() => {
    let cursor = 0;
    return draft.shots.map((shot, index) => {
      const start = cursor;
      cursor += Math.max(0, shot.duration_seconds);
      return { shot, index, start, end: cursor };
    });
  }, [draft.shots]);
  const total = ranges.at(-1)?.end ?? 0;
  const selected = ranges.find((range) => range.shot.id === selectedId) ?? ranges[0];
  const [playhead, setPlayhead] = useState(() => selected ? (selected.start + selected.end) / 2 : 0);
  const previewRef = useRef<HTMLVideoElement>(null);
  const [detectionSensitivity, setDetectionSensitivity] = useState<RV2VShotDetectionRequest["sensitivity"]>("medium");
  const [minShotFrames, setMinShotFrames] = useState(() => Math.max(4, Math.round(draft.render.fps / 2)));
  const [detectingShots, setDetectingShots] = useState(false);
  const [detectionMessage, setDetectionMessage] = useState<string | null>(null);
  const latestDraftRef = useRef(draft);
  const detectionRequestRef = useRef(0);
  latestDraftRef.current = draft;

  useEffect(() => () => {
    // A detector response may arrive after switching modes. Invalidate it so
    // an unmounted RV2V editor can never write an old draft back into App.
    detectionRequestRef.current += 1;
  }, []);

  useEffect(() => {
    setPlayhead((value) => Math.min(Math.max(value, 0), total));
  }, [total]);

  useEffect(() => {
    if (!selected) return;
    setPlayhead((value) =>
      value < selected.start || value > selected.end
        ? (selected.start + selected.end) / 2
        : value,
    );
  }, [selected?.shot.id, selected?.start, selected?.end]);

  const sourceTimeAt = (timelineTime: number) => {
    if (!selected) return 0;
    const outputDuration = Math.max(selected.shot.duration_seconds, 0.01);
    const progress = Math.min(
      1,
      Math.max(0, (timelineTime - selected.start) / outputDuration),
    );
    return selected.shot.source_start_seconds +
      selected.shot.source_duration_seconds * progress;
  };

  const syncPreviewToPlayhead = () => {
    const preview = previewRef.current;
    if (!preview || !selected?.shot.source_video) return;
    const sourceTime = sourceTimeAt(playhead);
    if (Number.isFinite(sourceTime) && Math.abs(preview.currentTime - sourceTime) > 0.03)
      preview.currentTime = sourceTime;
  };

  useEffect(syncPreviewToPlayhead, [
    playhead,
    selected?.start,
    selected?.shot.id,
    selected?.shot.source_start_seconds,
    selected?.shot.source_duration_seconds,
    selected?.shot.duration_seconds,
  ]);

  const select = (shotId: string) => {
    const range = ranges.find((item) => item.shot.id === shotId);
    if (!range) return;
    onSelected(shotId);
    setPlayhead((range.start + range.end) / 2);
  };
  const seekTimeline = (nextTime: number) => {
    const clamped = Math.min(Math.max(nextTime, 0), total);
    const containing = ranges.find((range, index) =>
      clamped >= range.start &&
      (clamped < range.end || (index === ranges.length - 1 && clamped <= range.end)),
    );
    if (containing && containing.shot.id !== selected?.shot.id)
      onSelected(containing.shot.id);
    setPlayhead(clamped);
  };
  const updateTimelineFromPreview = () => {
    const preview = previewRef.current;
    if (!preview || !selected) return;
    const sourceStart = selected.shot.source_start_seconds;
    const sourceDuration = Math.max(selected.shot.source_duration_seconds, 0.01);
    const sourceEnd = sourceStart + sourceDuration;
    if (preview.currentTime < sourceStart) {
      preview.currentTime = sourceStart;
      return;
    }
    if (preview.currentTime >= sourceEnd) {
      preview.currentTime = sourceEnd;
      preview.pause();
    }
    const progress = Math.min(1, Math.max(0, (preview.currentTime - sourceStart) / sourceDuration));
    setPlayhead(selected.start + selected.shot.duration_seconds * progress);
  };
  const detectShots = async () => {
    const source = selected?.shot.source_video;
    if (!selected || !source || detectingShots) return;
    const requestId = ++detectionRequestRef.current;
    const requested = {
      shotId: selected.shot.id,
      assetId: source.id,
      sourceStart: selected.shot.source_start_seconds,
      sourceDuration: selected.shot.source_duration_seconds,
      frameRate: draft.render.fps,
    };
    setDetectingShots(true);
    setDetectionMessage(null);
    try {
      const result = await directorApi.detectRV2VShots({
        asset_id: requested.assetId,
        frame_rate: requested.frameRate,
        sensitivity: detectionSensitivity,
        min_shot_frames: minShotFrames,
      });
      if (requestId !== detectionRequestRef.current) return;
      const latest = latestDraftRef.current;
      const latestShot = latest.shots.find((shot) => shot.id === requested.shotId);
      if (
        !latestShot ||
        latestShot.source_video?.id !== requested.assetId ||
        latestShot.source_start_seconds !== requested.sourceStart ||
        latestShot.source_duration_seconds !== requested.sourceDuration ||
        latest.render.fps !== requested.frameRate
      ) {
        setDetectionMessage("源区间、源视频或帧率已变化，本次检测结果已忽略");
        return;
      }
      const next = splitRV2VShotAtSourceCuts(
        latest,
        requested.shotId,
        result.cut_frames,
        requested.frameRate,
      );
      if (next === latest) {
        setDetectionMessage(latest.shots.length >= 128
          ? "时间线已达到 128 段上限，无法继续智能分镜"
          : "当前源截取范围内未检测到可分割的镜头切点");
        return;
      }
      onChange(next);
      const warning = result.warnings.length ? `；${result.warnings.join("；")}` : "";
      const limitWarning = next.shots.length >= 128
        ? "；已达到 128 段上限，超出的切点已忽略"
        : "";
      setDetectionMessage(`智能分镜完成：当前片段已拆为 ${next.shots.length - latest.shots.length + 1} 段${limitWarning}${warning}`);
    } catch (reason) {
      if (requestId === detectionRequestRef.current)
        setDetectionMessage(reason instanceof Error ? reason.message : "智能分镜失败");
    } finally {
      if (requestId === detectionRequestRef.current) setDetectingShots(false);
    }
  };
  const splitOffset = selected ? playhead - selected.start : 0;
  const canSplit = Boolean(
    selected?.shot.source_video &&
    draft.shots.length < 128 &&
    splitOffset > 0.01 &&
    splitOffset < (selected?.shot.duration_seconds ?? 0) - 0.01,
  );
  const selectedIndex = selected?.index ?? -1;

  return (
    <div className="rv2v-editor" aria-label="RV2V 可视时间线">
      <div className="rv2v-editor__preview">
        {selected?.shot.source_video?.preview_url ? (
          <video ref={previewRef} controls muted preload="metadata" key={`${selected.shot.id}:${selected.shot.source_video.id}`} src={selected.shot.source_video.preview_url} onLoadedMetadata={syncPreviewToPlayhead} onTimeUpdate={updateTimelineFromPreview} onPlay={syncPreviewToPlayhead} />
        ) : (
          <div className="rv2v-editor__empty"><span>RV2V</span><small>选择片段并上传源视频后可预览</small></div>
        )}
        <div><strong>{selected?.shot.title || "未选择片段"}</strong><small>{selected?.shot.source_video ? `${selected.shot.source_video.name} · 源 ${selected.shot.source_start_seconds.toFixed(2)}–${(selected.shot.source_start_seconds + selected.shot.source_duration_seconds).toFixed(2)}s` : "尚未绑定源视频"}</small></div>
      </div>
      <div className="rv2v-editor__toolbar">
        <button type="button" className="button button--ghost" disabled={selectedIndex <= 0} onClick={() => selected && onChange(moveRV2VShot(draft, selected.shot.id, -1))}>左移片段</button>
        <button type="button" className="button button--ghost" disabled={selectedIndex < 0 || selectedIndex >= draft.shots.length - 1} onClick={() => selected && onChange(moveRV2VShot(draft, selected.shot.id, 1))}>右移片段</button>
        <button type="button" className="button button--ghost" disabled={!canSplit} onClick={() => selected && onChange(splitRV2VShot(draft, selected.shot.id, splitOffset))}>在播放头分割</button>
        <button type="button" className="button button--danger" disabled={!selected || draft.shots.length <= 1} onClick={() => {
          if (!selected) return;
          const fallback = draft.shots[selected.index + 1] ?? draft.shots[selected.index - 1];
          if (fallback) onSelected(fallback.id);
          onChange(deleteRV2VShot(draft, selected.shot.id));
        }}>删除片段</button>
        <span>{playhead.toFixed(2)}s / {total.toFixed(2)}s</span>
      </div>
      <div className="rv2v-editor__detect">
        <label><span>检测灵敏度</span><select aria-label="智能分镜灵敏度" value={detectionSensitivity} disabled={detectingShots} onChange={(event) => setDetectionSensitivity(event.target.value as RV2VShotDetectionRequest["sensitivity"])}><option value="low">低</option><option value="medium">中</option><option value="high">高</option></select></label>
        <label><span>最短镜头帧数</span><DeferredNumberInput aria-label="智能分镜最短镜头帧数" min="4" max="100000" step="1" value={minShotFrames} disabled={detectingShots} normalizeValue={Math.trunc} onValueCommit={setMinShotFrames} /></label>
        <button type="button" className="button button--primary" disabled={!runtimeEnabled || detectingShots || draft.shots.length >= 128 || !selected?.shot.source_video || !Number.isFinite(draft.render.fps) || draft.render.fps < 1 || draft.render.fps > 240} onClick={() => void detectShots()}>{detectingShots ? "检测中…" : "智能分镜"}</button>
        {detectionMessage && <small role="status">{detectionMessage}</small>}
      </div>
      <input className="rv2v-editor__seek" aria-label="RV2V 播放头" type="range" min="0" max={Math.max(total, 0.01)} step="0.01" value={playhead} onChange={(event) => seekTimeline(Number(event.target.value))} />
      <div className="rv2v-editor__track" role="group" aria-label="RV2V 片段轨道">
        {ranges.map(({ shot, index, start, end }) => <button type="button" aria-pressed={shot.id === selected?.shot.id} className={shot.id === selected?.shot.id ? "is-selected" : ""} style={{ flexGrow: Math.max(shot.duration_seconds, 0.01) }} key={shot.id} onClick={() => select(shot.id)}><strong>{String(index + 1).padStart(2, "0")} · {shot.title}</strong><small>{start.toFixed(2)}–{end.toFixed(2)}s</small><em>{shot.source_video?.name ?? "等待源视频"}</em></button>)}
      </div>
    </div>
  );
}

function ModeShotInputs({
  draft,
  index,
  runtimeEnabled,
  onChange,
  onAssetsChange,
}: TimelineEditorProps & { index: number }) {
  switch (draft.mode) {
    case "t2v":
      return <p className="shot-assets-note">此镜头由提示词直接生成，无需素材。</p>;
    case "i2v": {
      const shot = draft.shots[index];
      return (
        <AssetUploader label="起始帧" description="仅用于当前镜头" kind="image" accept="image/png,image/jpeg,image/webp"
          disabled={!runtimeEnabled}
          assets={shot.first_image ? [shot.first_image] : []}
          onMutation={(mutation) => onAssetsChange(draft.mode, shot.id, "first_image", mutation)} />
      );
    }
    case "fl2v": {
      const shot = draft.shots[index];
      return (
        <div className="asset-grid">
          <AssetUploader label="首帧" description="首尾至少上传一张" kind="image" accept="image/png,image/jpeg,image/webp"
            disabled={!runtimeEnabled}
            assets={shot.first_image ? [shot.first_image] : []}
            onMutation={(mutation) => onAssetsChange(draft.mode, shot.id, "first_image", mutation)} />
          <AssetUploader label="尾帧" description="首尾至少上传一张" kind="image" accept="image/png,image/jpeg,image/webp"
            disabled={!runtimeEnabled}
            assets={shot.last_image ? [shot.last_image] : []}
            onMutation={(mutation) => onAssetsChange(draft.mode, shot.id, "last_image", mutation)} />
        </div>
      );
    }
    case "r2v": {
      const shot = draft.shots[index];
      return (
        <div className="asset-grid asset-grid--three">
          <AssetUploader label="参考图" description={`本组图片 1–${MINIMAX_H3_REFERENCE_LIMITS.referenceImages}`} kind="image" accept="image/png,image/jpeg,image/webp" multiple maxItems={MINIMAX_H3_REFERENCE_LIMITS.referenceImages}
            disabled={!runtimeEnabled}
            slotted
            assets={shot.reference_images}
            onMutation={(mutation) => onAssetsChange(draft.mode, shot.id, "reference_images", mutation)} />
          <AssetUploader label="参考音频" description={`本组音频 1–${MINIMAX_H3_REFERENCE_LIMITS.referenceAudios}`} kind="audio" accept=".wav,.mp3,.flac,.ogg,.oga,.m4a,.aac" multiple maxItems={MINIMAX_H3_REFERENCE_LIMITS.referenceAudios}
            disabled={!runtimeEnabled}
            slotted
            assets={shot.reference_audios}
            onMutation={(mutation) => onAssetsChange(draft.mode, shot.id, "reference_audios", mutation)} />
          <AssetUploader label="参考视频" description={`本组视频 1–${MINIMAX_H3_REFERENCE_LIMITS.totalReferenceVideos}`} kind="video" accept=".mp4,.m4v,.mov,.webm,.mkv,.avi,.mpeg,.mpg" multiple maxItems={MINIMAX_H3_REFERENCE_LIMITS.totalReferenceVideos}
            disabled={!runtimeEnabled}
            slotted
            assets={shot.reference_videos}
            onMutation={(mutation) => onAssetsChange(draft.mode, shot.id, "reference_videos", mutation)} />
        </div>
      );
    }
    case "v2v": {
      const shot = draft.shots[index];
      return (
        <div className="source-segment-grid">
          <AssetUploader label="源视频" description="当前片段的运动与时序基底" kind="video" accept=".mp4,.m4v,.mov,.webm,.mkv,.avi,.mpeg,.mpg"
            disabled={!runtimeEnabled}
            assets={shot.source_video ? [shot.source_video] : []}
            onMutation={(mutation) => onAssetsChange(draft.mode, shot.id, "source_video", mutation)} />
          <div className="field-grid field-grid--two">
            <Field label="源起点（秒）"><DeferredNumberInput min="0" max={shot.source_video?.metadata?.duration ?? 86_400} step="0.1" value={shot.source_start_seconds}
              onValueCommit={(value) => onChange({ ...draft, shots: draft.shots.map((item, i) => i === index ? { ...item, source_start_seconds: value } : item) })} /></Field>
            <Field label="源时长（秒）"><DeferredNumberInput min="0.01" max={shot.source_video?.metadata ? Math.max(0, shot.source_video.metadata.duration - shot.source_start_seconds) : 86_400} step="0.01" value={shot.source_duration_seconds}
              onValueCommit={(value) => onChange({ ...draft, shots: draft.shots.map((item, i) => i === index ? { ...item, source_duration_seconds: value } : item) })} /></Field>
          </div>
        </div>
      );
    }
    case "rv2v": {
      const shot = draft.shots[index];
      return (
        <div className="source-segment-grid">
          <div className="asset-grid asset-grid--three">
            <AssetUploader label="源视频" description="当前片段，必选" kind="video" accept=".mp4,.m4v,.mov,.webm,.mkv,.avi,.mpeg,.mpg"
              disabled={!runtimeEnabled}
              assets={shot.source_video ? [shot.source_video] : []}
              onMutation={(mutation) => onAssetsChange(draft.mode, shot.id, "source_video", mutation)} />
            <AssetUploader label="参考图" description="当前片段图片参考" kind="image" accept="image/png,image/jpeg,image/webp" multiple maxItems={MINIMAX_H3_REFERENCE_LIMITS.referenceImages}
              disabled={!runtimeEnabled}
              slotted
              assets={shot.reference_images}
              onMutation={(mutation) => onAssetsChange(draft.mode, shot.id, "reference_images", mutation)} />
            <AssetUploader label="参考音频" description="当前片段声音参考" kind="audio" accept=".wav,.mp3,.flac,.ogg,.oga,.m4a,.aac" multiple maxItems={MINIMAX_H3_REFERENCE_LIMITS.referenceAudios}
              disabled={!runtimeEnabled}
              slotted
              assets={shot.reference_audios}
              onMutation={(mutation) => onAssetsChange(draft.mode, shot.id, "reference_audios", mutation)} />
          </div>
          <div className="field-grid field-grid--two">
            <Field label="源起点（秒）"><DeferredNumberInput min="0" max={shot.source_video?.metadata?.duration ?? 86_400} step="0.1" value={shot.source_start_seconds}
              onValueCommit={(value) => onChange({ ...draft, shots: draft.shots.map((item, i) => i === index ? { ...item, source_start_seconds: value } : item) })} /></Field>
            <Field label="源时长（秒）"><DeferredNumberInput min="0.01" max={shot.source_video?.metadata ? Math.max(0, shot.source_video.metadata.duration - shot.source_start_seconds) : 86_400} step="0.01" value={shot.source_duration_seconds}
              onValueCommit={(value) => onChange({ ...draft, shots: draft.shots.map((item, i) => i === index ? { ...item, source_duration_seconds: value } : item) })} /></Field>
          </div>
        </div>
      );
    }
  }
}

export function TimelineEditor({ draft, runtimeEnabled, onChange, onAssetsChange }: TimelineEditorProps) {
  const total = effectiveDraftTiming(draft);
  const [selectedShotId, setSelectedShotId] = useState(draft.shots[0]?.id ?? "");
  useEffect(() => {
    if (!draft.shots.some((shot) => shot.id === selectedShotId))
      setSelectedShotId(draft.shots[0]?.id ?? "");
  }, [draft.shots, selectedShotId]);
  return (
    <Panel eyebrow="编排 03" title={draft.mode === "r2v" ? "参考组" : draft.mode === "v2v" || draft.mode === "rv2v" ? "源视频时间线" : "镜头时间线"}
      description="素材和参数归属于各自镜头；复制或切换模式不会串值。"
      action={<div className="timeline-total"><strong>{total.frames}f</strong><small>{total.durationSeconds > 0 ? `${total.durationSeconds.toFixed(2)}s` : "—"} · {draft.shots.length} 段</small></div>}
      className="timeline-panel">
      {draft.mode === "rv2v" && <RV2VTimelineTrack draft={draft} selectedId={selectedShotId} runtimeEnabled={runtimeEnabled} onSelected={setSelectedShotId} onChange={onChange} />}
      <div className="timeline-ruler" aria-hidden="true"><span>00:00</span><i /><span>{total.frames}f / {total.durationSeconds.toFixed(2)}s</span></div>
      <div className="shot-list">
        {draft.shots.map((shot, index) => (
          <article className={`shot-card ${shot.enabled ? "" : "is-disabled"} ${draft.mode === "rv2v" && shot.id === selectedShotId ? "is-selected" : ""}`} key={shot.id}>
            <header className="shot-card__head">
              <span className="shot-card__number">{String(index + 1).padStart(2, "0")}</span>
              <input aria-label={`镜头 ${index + 1} 名称`} maxLength={256} value={shot.title}
                onChange={(event) => onChange(replaceBaseShot(draft, index, { title: event.target.value }))} />
              <label className="toggle"><input type="checkbox" checked={shot.enabled}
                onChange={(event) => onChange(replaceBaseShot(draft, index, { enabled: event.target.checked }))} /><span />启用</label>
              <button type="button" className="icon-button" aria-label={`删除镜头 ${index + 1}`} disabled={draft.shots.length === 1}
                onClick={() => onChange(removeShot(draft, index))}>×</button>
            </header>
            <div className="shot-card__body">
              <div className="shot-copy-grid">
                <Field label="镜头提示词（覆盖）" hint="非空时覆盖默认提示词；留空则继承默认提示词"><textarea rows={3} value={shot.prompt} placeholder="动作、构图、运镜…"
                  onChange={(event) => onChange(replaceBaseShot(draft, index, { prompt: limitPromptCharacters(event.target.value) }))} /></Field>
                <Field label="生成时长（秒）" hint={(() => { const timing = effectiveShotTiming(shot.duration_seconds, draft.render.fps); return timing.frames > 0 ? `输入 ${shot.duration_seconds}s → ${timing.frames} 帧 · 实际 ${timing.durationSeconds.toFixed(2)}s` : "请输入有效时长与帧率"; })()}><DeferredNumberInput aria-label="生成时长（秒）" min={draft.mode === "fl2v" ? "0.1" : "0.01"} max="120" step={draft.mode === "fl2v" ? "0.1" : "0.01"} value={shot.duration_seconds}
                  onValueCommit={(value) => onChange(replaceBaseShot(draft, index, { duration_seconds: value }))} /></Field>
              </div>
              <ModeShotInputs draft={draft} index={index} runtimeEnabled={runtimeEnabled} onChange={onChange} onAssetsChange={onAssetsChange} />
            </div>
          </article>
        ))}
      </div>
      <button type="button" className="add-shot" disabled={draft.shots.length >= 128} onClick={() => onChange(addShot(draft))}><span>＋</span>{draft.shots.length >= 128 ? "已达到 128 段上限" : `添加${draft.mode === "r2v" ? "参考组" : "镜头"}`}</button>
    </Panel>
  );
}
