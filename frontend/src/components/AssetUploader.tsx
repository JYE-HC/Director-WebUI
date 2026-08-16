import { useRef, useState } from "react";
import { directorApi } from "../api/client";
import {
  describeUploadProgress,
  uploadClassifiedDroppedFiles,
  type DroppedUploadProgress,
} from "../domain/assetDrag";
import type { AssetKind, AssetReference, SlottedAssetReference } from "../domain/modes";
import type { AssetMutation } from "../state/directorState";
import { Spinner } from "./ui";

interface AssetUploaderBaseProps {
  label: string;
  description: string;
  kind: AssetKind;
  accept: string;
  multiple?: boolean;
  maxItems?: number;
  disabled?: boolean;
}

interface PlainAssetUploaderProps extends AssetUploaderBaseProps {
  slotted?: false;
  assets: AssetReference[];
  onMutation: (mutation: AssetMutation) => void;
}

interface SlottedAssetUploaderProps extends AssetUploaderBaseProps {
  slotted: true;
  assets: SlottedAssetReference[];
  onMutation: (mutation: AssetMutation) => void;
}

type AssetUploaderProps = PlainAssetUploaderProps | SlottedAssetUploaderProps;

function referenceTag(asset: AssetReference): string | null {
  if (!("slot" in asset) || !Number.isInteger(asset.slot)) return null;
  const label = asset.kind === "image" ? "Picture" : asset.kind === "audio" ? "Audio" : "Video";
  return `<${label} ${Number(asset.slot) + 1}>`;
}

export function AssetUploader(props: AssetUploaderProps) {
  const {
  label,
  description,
  kind,
  accept,
  assets,
  multiple = false,
  maxItems = 1,
  disabled = false,
  } = props;
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<DroppedUploadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  const upload = async (files: FileList | null) => {
    if (disabled || !files?.length) return;
    const remaining = Math.max(0, maxItems - (multiple ? assets.length : 0));
    const selected = Array.from(files).slice(0, multiple ? remaining : 1);
    if (!selected.length) {
      setError(`最多允许 ${maxItems} 个素材`);
      return;
    }
    setUploading(true);
    setError(null);
    try {
      const result = await uploadClassifiedDroppedFiles(
        selected.map((file) => ({ file, kind })),
        (file, assetKind, report) => directorApi.uploadAsset(file, assetKind, report),
        () => true,
        setUploadProgress,
      );
      const uploaded: AssetReference[] = result.assets;
      // Send only the completed upload delta. The parent reducer merges it into
      // the latest mode/shot state, so edits made while the request was in
      // flight cannot be overwritten by this component's render-time props.
      if (uploaded.length) props.onMutation({ type: "add", assets: uploaded });
      if (result.failures.length) {
        setError(result.failures.map((failure) => `${failure.file_name}：${failure.message}`).join("；"));
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "素材上传失败");
    } finally {
      setUploading(false);
      setUploadProgress(null);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  return (
    <div className="asset-field">
      <div className="asset-field__topline">
        <div>
          <strong>{label}</strong>
          <small>{description}</small>
        </div>
        <span>{multiple ? `${assets.length}/${maxItems}` : assets.length ? "已就绪" : "必选"}</span>
      </div>
      <label className={`drop-zone ${uploading ? "is-uploading" : ""} ${disabled ? "is-disabled" : ""}`}>
        <input
          ref={inputRef}
          type="file"
          aria-label={label}
          accept={accept}
          multiple={multiple}
          disabled={disabled || uploading || (multiple && assets.length >= maxItems)}
          onChange={(event) => void upload(event.target.files)}
        />
        {disabled ? (
          <>
            <span className="drop-zone__plus">—</span>
            <span>配置 ComfyUI 地址并等待连接就绪后可上传素材</span>
          </>
        ) : uploading ? (
          <>
            <Spinner label={`正在上传${label}`} />
            <span>{uploadProgress ? describeUploadProgress(uploadProgress) : "上传到素材库…"}</span>
          </>
        ) : (
          <>
            <span className="drop-zone__plus">＋</span>
            <span>点击选择或拖入文件</span>
            <small>{accept.replaceAll(",", " · ")}</small>
          </>
        )}
      </label>
      {error && <p className="inline-error" role="alert">{error}</p>}
      {assets.length > 0 && (
        <ul className="asset-list">
          {assets.map((asset) => (
            <li key={`${asset.id}-${"slot" in asset ? asset.slot : "plain"}`}>
              {asset.kind === "image" && asset.preview_url ? (
                <img src={asset.preview_url} alt="" />
              ) : (
                <span className="asset-list__glyph" aria-hidden="true">
                  {asset.kind === "video" ? "▶" : asset.kind === "audio" ? "♫" : "▧"}
                </span>
              )}
              <span className="asset-list__name">
                <strong title={asset.name}>{referenceTag(asset) && <em>{referenceTag(asset)}</em>}{asset.name}</strong>
                <small>
                  {asset.kind === "video" && asset.metadata
                    ? `${asset.metadata.duration.toFixed(2)}s · ${asset.metadata.native_fps.toFixed(2)}fps · ${asset.metadata.frame_count}帧 · ${asset.metadata.width}×${asset.metadata.height}`
                    : asset.subfolder || asset.type}
                </small>
              </span>
              <button
                type="button"
                aria-label={`移除 ${asset.name}`}
                onClick={() => props.onMutation({ type: "remove", assetId: asset.id })}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
