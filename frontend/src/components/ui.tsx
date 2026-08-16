import {
  useEffect,
  useRef,
  useState,
  type ComponentPropsWithoutRef,
  type PropsWithChildren,
  type ReactNode,
} from "react";

type DeferredNumberInputProps = Omit<
  ComponentPropsWithoutRef<"input">,
  "type" | "value" | "defaultValue" | "onChange"
> & {
  value: number;
  onValueCommit: (value: number) => void;
  normalizeValue?: (value: number) => number;
};

function finiteNumberAttribute(value: string | number | undefined): number | undefined {
  if (value === undefined) return undefined;
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

/**
 * Keeps the user's in-progress text local, then applies numeric bounds only
 * when editing finishes. This permits intermediate values such as an empty
 * field or `1` while the user is entering `12` into a field whose minimum is 4.
 */
export function DeferredNumberInput({
  value,
  onValueCommit,
  normalizeValue,
  min,
  max,
  onBlur,
  onKeyDown,
  ...props
}: DeferredNumberInputProps) {
  const externalText = Number.isFinite(value) ? String(value) : "";
  const [draft, setDraft] = useState(externalText);
  const cancelOnBlurRef = useRef(false);

  useEffect(() => setDraft(externalText), [externalText]);

  const commit = (raw: string) => {
    const trimmed = raw.trim();
    const parsed = Number(trimmed);
    if (!trimmed || !Number.isFinite(parsed)) {
      setDraft(externalText);
      return;
    }

    let next = normalizeValue ? normalizeValue(parsed) : parsed;
    if (!Number.isFinite(next)) {
      setDraft(externalText);
      return;
    }
    const lower = finiteNumberAttribute(min);
    const upper = finiteNumberAttribute(max);
    if (lower !== undefined) next = Math.max(lower, next);
    if (upper !== undefined) next = Math.min(upper, next);

    if (Object.is(next, value)) {
      setDraft(String(next));
      return;
    }
    onValueCommit(next);
    // The parent may normalize the committed value, including back to the
    // same authoritative value it already had. Return to that authority until
    // the controlled prop reports what was accepted; otherwise an unchanged
    // externalText dependency cannot clear the now-stale local draft.
    setDraft(externalText);
  };

  return <input
    {...props}
    type="number"
    min={min}
    max={max}
    value={draft}
    onChange={(event) => setDraft(event.currentTarget.value)}
    onBlur={(event) => {
      if (cancelOnBlurRef.current) {
        cancelOnBlurRef.current = false;
        setDraft(externalText);
      } else {
        commit(event.currentTarget.value);
      }
      onBlur?.(event);
    }}
    onKeyDown={(event) => {
      onKeyDown?.(event);
      if (event.defaultPrevented) return;
      if (event.key === "Enter") {
        event.preventDefault();
        event.currentTarget.blur();
      } else if (event.key === "Escape") {
        event.preventDefault();
        cancelOnBlurRef.current = true;
        event.currentTarget.blur();
      }
    }}
  />;
}

export function Panel({
  eyebrow,
  title,
  description,
  action,
  className = "",
  children,
}: PropsWithChildren<{
  eyebrow?: string;
  title: string;
  description?: string;
  action?: ReactNode;
  className?: string;
}>) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel__head">
        <div>
          {eyebrow && <span className="eyebrow">{eyebrow}</span>}
          <h2>{title}</h2>
          {description && <p>{description}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

export function Field({
  label,
  hint,
  children,
  className = "",
}: PropsWithChildren<{ label: string; hint?: string; className?: string }>) {
  return (
    <label className={`field ${className}`}>
      <span className="field__label">{label}</span>
      {children}
      {hint && <small>{hint}</small>}
    </label>
  );
}

export function StatusDot({ state }: { state: "online" | "offline" | "checking" | "unknown" }) {
  return <span className={`status-dot status-dot--${state}`} aria-hidden="true" />;
}

export function Spinner({ label = "处理中" }: { label?: string }) {
  return <span className="spinner" role="status" aria-label={label} />;
}

export function formatBytes(bytes?: number): string {
  if (bytes === undefined) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 10 || unit === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

export function formatTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}
