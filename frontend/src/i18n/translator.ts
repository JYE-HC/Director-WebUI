import zhCN from "./locales/zh-CN.json";

export const DEFAULT_LOCALE = "zh-CN";

export type TranslationParam = string | number;
export type TranslationParams = Readonly<Record<string, TranslationParam>>;
export type BundledTranslationKey = keyof typeof zhCN;
export type TranslationKey = BundledTranslationKey | (string & {});
export type MessageCatalog = Readonly<Record<string, string>>;
export type MessageCatalogs = Readonly<Record<string, MessageCatalog>>;

export interface Translator {
  readonly locale: string;
  readonly fallbackLocale: string;
  t(key: TranslationKey, params?: TranslationParams, fallback?: string): string;
}

export interface TranslatorOptions {
  locale?: string;
  fallbackLocale?: string;
  catalogs?: MessageCatalogs;
}

export const BUNDLED_CATALOGS: MessageCatalogs = Object.freeze({
  [DEFAULT_LOCALE]: zhCN,
});

function catalogMessage(catalog: MessageCatalog | undefined, key: string): string | undefined {
  if (!catalog || !Object.hasOwn(catalog, key)) return undefined;
  const value = catalog[key];
  return value.length > 0 ? value : undefined;
}

function interpolate(template: string, params: TranslationParams | undefined): string {
  if (!params) return template;
  return template.replace(/\{([A-Za-z][A-Za-z0-9_]*)\}/g, (placeholder, name: string) =>
    Object.hasOwn(params, name) ? String(params[name]) : placeholder,
  );
}

export function createTranslator(options: TranslatorOptions = {}): Translator {
  const locale = options.locale ?? DEFAULT_LOCALE;
  const fallbackLocale = options.fallbackLocale ?? DEFAULT_LOCALE;
  const catalogs: MessageCatalogs = options.catalogs
    ? { ...BUNDLED_CATALOGS, ...options.catalogs }
    : BUNDLED_CATALOGS;

  return Object.freeze({
    locale,
    fallbackLocale,
    t(key: TranslationKey, params?: TranslationParams, fallback?: string): string {
      const template = catalogMessage(catalogs[locale], key)
        ?? (fallbackLocale === locale ? undefined : catalogMessage(catalogs[fallbackLocale], key))
        ?? fallback
        ?? key;
      return interpolate(template, params);
    },
  });
}

export const defaultTranslator = createTranslator();
