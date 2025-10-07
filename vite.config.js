import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
  },
  // This is the fix for the MediaPipe module loading issue.
  // It tells Vite to treat the MediaPipe libraries as if they have a default export.
  resolve: {
    alias: {
      '@mediapipe/camera_utils': '@mediapipe/camera_utils/camera_utils.js',
      '@mediapipe/hands': '@mediapipe/hands/hands.js',
    },
  },
});
