interface Props {
  pct: number;
}

function barColor(pct: number): string {
  if (pct >= 80) return 'bg-green-500';
  if (pct >= 60) return 'bg-yellow-400';
  return 'bg-red-500';
}

export default function ConfidenceBar({ pct }: Props) {
  const color = barColor(pct);
  return (
    <div className="flex items-center gap-3">
      <div className="flex-1 h-3 bg-slate-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${color}`}
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
      <span className="text-lg text-slate-600 font-medium w-16 text-right">
        {pct.toFixed(1)}%
      </span>
    </div>
  );
}
