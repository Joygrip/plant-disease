import type { PredictionResponse } from '../types';
import ConfidenceBar from './ConfidenceBar';
import TopKList from './TopKList';
import WarningBanner from './WarningBanner';

interface Props {
  result: PredictionResponse;
  onReset: () => void;
}

export default function ResultCard({ result, onReset }: Props) {
  const modelName = result.model_used ?? result.model;

  return (
    <div className="w-full space-y-4">
      <div>
        <p className="text-sm text-slate-500 uppercase tracking-wide mb-1">Diagnosis</p>
        <h2 className="text-2xl font-semibold text-slate-800">{result.prediction_display}</h2>
      </div>

      <div>
        <p className="text-sm text-slate-500 mb-1">Confidence</p>
        <ConfidenceBar pct={result.confidence_pct} />
      </div>

      {result.warning && <WarningBanner message={result.warning} />}

      {result.top_k.length > 1 && (
        <div>
          <p className="text-sm text-slate-500 mb-2">Top predictions</p>
          <TopKList items={result.top_k} />
        </div>
      )}

      <p className="text-xs text-slate-400">
        Model: {modelName} · {result.inference_ms.toFixed(1)} ms
      </p>

      <button
        type="button"
        onClick={onReset}
        className="w-full py-2 px-4 rounded-lg border border-slate-300 text-slate-600
          hover:bg-slate-50 transition-colors text-sm font-medium"
      >
        Diagnose another
      </button>
    </div>
  );
}
