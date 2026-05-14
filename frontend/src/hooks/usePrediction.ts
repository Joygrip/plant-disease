import { useState } from 'react';
import axios from 'axios';
import { predict } from '../api/client';
import type { PredictionResponse } from '../types';

type State =
  | { status: 'idle' }
  | { status: 'selected'; file: File; previewUrl: string }
  | { status: 'loading'; file: File; previewUrl: string }
  | { status: 'result'; file: File; previewUrl: string; result: PredictionResponse }
  | { status: 'error'; file: File; previewUrl: string; message: string };

function errorMessage(err: unknown): string {
  if (axios.isAxiosError(err) && err.response?.data?.error?.message) {
    return err.response.data.error.message as string;
  }
  if (err instanceof Error) return err.message;
  return 'An unexpected error occurred';
}

export function usePrediction() {
  const [state, setState] = useState<State>({ status: 'idle' });

  const selectFile = (file: File) => {
    const previewUrl = URL.createObjectURL(file);
    setState({ status: 'selected', file, previewUrl });
  };

  const reset = () => {
    if (state.status !== 'idle') {
      URL.revokeObjectURL((state as { previewUrl: string }).previewUrl);
    }
    setState({ status: 'idle' });
  };

  const retrySelect = () => {
    if (state.status === 'error') {
      setState({ status: 'selected', file: state.file, previewUrl: state.previewUrl });
    }
  };

  const diagnose = async () => {
    if (state.status !== 'selected') return;
    const { file, previewUrl } = state;
    setState({ status: 'loading', file, previewUrl });
    try {
      const result = await predict(file);
      setState({ status: 'result', file, previewUrl, result });
    } catch (err) {
      setState({ status: 'error', file, previewUrl, message: errorMessage(err) });
    }
  };

  return { state, selectFile, reset, retrySelect, diagnose };
}
