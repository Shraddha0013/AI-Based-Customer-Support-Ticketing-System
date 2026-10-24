import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { DEFAULT: "#14232E", soft: "#475866", mute: "#76868F" },
        paper: "#F2F4F6",
        line: "#DAE1E6",
        brand: { DEFAULT: "#0E6B6B", dark: "#0A4F4F", tint: "#DFF0EE" },
        signal: { ok: "#1F7A4D", warn: "#B86A00", bad: "#B42318" },
      },
    },
  },
  plugins: [],
};
export default config;
