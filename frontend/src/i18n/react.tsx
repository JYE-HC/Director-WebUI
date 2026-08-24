import { createContext, useContext, useMemo, type ReactNode } from "react";
import {
  createTranslator,
  defaultTranslator,
  type MessageCatalogs,
  type Translator,
} from "./translator";

const I18nContext = createContext<Translator>(defaultTranslator);

export interface I18nProviderProps {
  children: ReactNode;
  locale?: string;
  fallbackLocale?: string;
  catalogs?: MessageCatalogs;
}

export function I18nProvider({
  children,
  locale,
  fallbackLocale,
  catalogs,
}: I18nProviderProps) {
  const translator = useMemo(
    () => createTranslator({ locale, fallbackLocale, catalogs }),
    [locale, fallbackLocale, catalogs],
  );
  return <I18nContext.Provider value={translator}>{children}</I18nContext.Provider>;
}

export function useTranslator(): Translator {
  return useContext(I18nContext);
}
