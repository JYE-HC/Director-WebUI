import { useCallback, useEffect, useRef, useState } from "react";

import { directorApi } from "../api/client";
import type {
  CapabilityReport,
  ComfyKitchenAttentionCapability,
  RuntimeSettings,
} from "../api/types";
import type { ComfyKitchenAttentionUiCapability } from "../components/ComfyKitchenAttentionField";
import {
  TIMELINE_MODE_ORDER,
  type TimelineGenerationMode,
} from "../domain/timelineProject";

const CHECKING: ComfyKitchenAttentionUiCapability = {
  state: "checking",
  backend: null,
  reasons: [],
};

function reachableFamilies(
  modes: readonly TimelineGenerationMode[],
): TimelineGenerationMode[] {
  return TIMELINE_MODE_ORDER.filter((family) => modes.includes(family));
}

export function comfyKitchenAttentionSettingsContextKey(
  settings: RuntimeSettings,
  families: readonly TimelineGenerationMode[],
): string {
  if (!settings.multi_gpu_enabled) {
    return JSON.stringify([
      "standard",
      ...families.map((family) => [family, settings.placement[family].device]),
    ]);
  }
  return JSON.stringify([
    "raylight",
    ...families.map((family) => {
      const profile = settings.placement[family].raylight;
      return [
        family,
        profile.ring_degree,
        [...profile.gpu_select].sort((left, right) => left - right),
      ];
    }),
  ]);
}

export function useComfyKitchenAttentionCapability({
  active,
  familyModes,
  connection,
  confirmedSettings,
  draftSettings,
}: {
  active: boolean;
  familyModes: readonly TimelineGenerationMode[];
  connection: CapabilityReport["connection"];
  confirmedSettings: RuntimeSettings;
  draftSettings: RuntimeSettings;
}): {
  capability: ComfyKitchenAttentionUiCapability;
  refreshHostCapability: () => void;
} {
  const families = reachableFamilies(familyModes);
  const familyKey = families.join(",");
  const confirmedContext = comfyKitchenAttentionSettingsContextKey(
    confirmedSettings,
    families,
  );
  const draftContext = comfyKitchenAttentionSettingsContextKey(draftSettings, families);
  const draftPending = confirmedContext !== draftContext;
  const [hostEpoch, setHostEpoch] = useState(0);
  const [resolved, setResolved] = useState<{
    key: string;
    value: ComfyKitchenAttentionCapability;
  } | null>(null);
  const resolvedRef = useRef(resolved);
  const requestGeneration = useRef(0);
  const requestKey = JSON.stringify([
    familyKey,
    confirmedContext,
    connection,
    hostEpoch,
  ]);

  const refreshHostCapability = useCallback(() => {
    setHostEpoch((current) => current + 1);
  }, []);

  useEffect(() => {
    if (!active || draftPending) return;
    const cached = resolvedRef.current;
    if (cached?.key === requestKey) {
      setResolved((previous) => previous === cached ? previous : cached);
      return;
    }
    const controller = new AbortController();
    const generation = ++requestGeneration.current;
    let current = true;
    setResolved((previous) => previous?.value.state === "unknown" ? null : previous);
    void directorApi.getComfyKitchenAttentionCapability(families, controller.signal)
      .then((value) => {
        if (!current || requestGeneration.current !== generation) return;
        const next = { key: requestKey, value };
        if (value.state !== "unknown") resolvedRef.current = next;
        setResolved(next);
      })
      .catch(() => {
        if (!current || requestGeneration.current !== generation) return;
        const value: ComfyKitchenAttentionCapability = {
          context_revision: "frontend:unavailable",
          backend: null,
          state: "unknown",
          reasons: [{
            code: "capability_request_failed",
            message: "暂时无法确认当前环境；仍可启用，实际执行结果由 ComfyUI 决定。",
          }],
        };
        const next = { key: requestKey, value };
        setResolved(next);
      });
    return () => {
      current = false;
      controller.abort();
    };
  }, [active, draftPending, familyKey, requestKey]);

  return {
    capability: active && !draftPending && resolved?.key === requestKey
      ? resolved.value
      : CHECKING,
    refreshHostCapability,
  };
}
