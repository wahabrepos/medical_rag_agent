"use client";

import { useState } from "react";

interface Props {
  apiUrl: string | undefined;
  mock: boolean;
  apiKey: string;
  onApiKey: (key: string) => void;
}

export function Settings({ apiUrl, mock, apiKey, onApiKey }: Props) {
  const [open, setOpen] = useState(false);
  return (
    <div className="settings">
      <button
        type="button"
        className="button secondary small"
        aria-expanded={open}
        aria-controls="settings-panel"
        onClick={() => setOpen(!open)}
      >
        Settings
      </button>
      {open && (
        <div id="settings-panel" className="settings-panel card">
          <p>
            <span className="label">Connected to</span>{" "}
            {mock ? "Demo mode (recorded answers, no API)" : apiUrl}
          </p>
          {!mock && (
            <label className="field">
              <span className="label">API key</span>
              <input
                type="password"
                autoComplete="off"
                value={apiKey}
                onChange={(e) => onApiKey(e.target.value)}
                placeholder="Bearer key issued for this API"
              />
              <span className="hint">Kept only in this browser.</span>
            </label>
          )}
        </div>
      )}
    </div>
  );
}
