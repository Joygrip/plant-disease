import type { ClassItem } from '../types';

interface Props {
  items: ClassItem[];
}

export default function TopKList({ items }: Props) {
  const max = items[0]?.probability ?? 1;

  return (
    <ol className="space-y-2">
      {items.map((item, idx) => (
        <li key={item.class} className="flex items-center gap-3">
          <span className="text-sm text-slate-400 w-4 shrink-0">{idx + 1}.</span>
          <div className="flex-1 min-w-0">
            <div className="flex justify-between mb-0.5">
              <span className="text-sm text-slate-700 truncate">{item.display}</span>
              <span className="text-sm text-slate-500 ml-2 shrink-0">
                {(item.probability * 100).toFixed(1)}%
              </span>
            </div>
            <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-slate-400 rounded-full"
                style={{ width: `${(item.probability / max) * 100}%` }}
              />
            </div>
          </div>
        </li>
      ))}
    </ol>
  );
}
