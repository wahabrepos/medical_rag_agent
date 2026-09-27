// Per-browser convenience storage; may be unavailable (private mode, blocked storage).
export function load(key: string): string {
  try {
    return window.localStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
}

export function save(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // ignore: the value simply is not remembered
  }
}
