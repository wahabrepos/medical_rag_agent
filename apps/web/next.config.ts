import type { NextConfig } from "next";

// Static export: the UI is plain files (Cloudflare Pages) that call the API from the browser.
const config: NextConfig = {
  output: "export",
  reactStrictMode: true,
  poweredByHeader: false,
};

export default config;
