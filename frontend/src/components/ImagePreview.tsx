interface Props {
  previewUrl: string;
  fileName: string;
  fileSize: number;
  onRemove: () => void;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function ImagePreview({ previewUrl, fileName, fileSize, onRemove }: Props) {
  return (
    <div className="w-full">
      <img
        src={previewUrl}
        alt="Selected leaf"
        className="max-h-[300px] w-full object-contain rounded-lg border border-slate-200"
      />
      <div className="flex items-center justify-between mt-2">
        <p className="text-sm text-slate-500 truncate max-w-xs">
          {fileName} <span className="text-slate-400">({formatSize(fileSize)})</span>
        </p>
        <button
          type="button"
          onClick={onRemove}
          className="text-sm text-slate-400 hover:text-red-500 transition-colors ml-2 shrink-0"
        >
          Remove
        </button>
      </div>
    </div>
  );
}
