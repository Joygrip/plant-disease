import axios from 'axios';
import type { PredictionResponse } from '../types';

const http = axios.create({ baseURL: '/api' });

export async function predict(
  file: File,
  model: string = 'mobilenet_v2',
  topK: number = 3,
): Promise<PredictionResponse> {
  const form = new FormData();
  form.append('image', file);
  form.append('model', model);

  const { data } = await http.post<PredictionResponse>(
    `/predict?top_k=${topK}`,
    form,
    { headers: { 'Content-Type': 'multipart/form-data' } },
  );
  return data;
}
