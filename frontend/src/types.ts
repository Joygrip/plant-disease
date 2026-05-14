export interface ClassItem {
  class: string;
  display: string;
  probability: number;
}

export interface PredictionResponse {
  prediction: string;
  prediction_display: string;
  confidence: number;
  confidence_pct: number;
  top_k: ClassItem[];
  inference_ms: number;
  model_used: string;
  model: string;
  image_size: { width: number; height: number };
  warning: string | null;
}

export interface ApiError {
  error: {
    code: string;
    message: string;
    detail: string;
  };
}
