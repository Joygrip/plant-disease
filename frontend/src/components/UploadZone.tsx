import { useCallback, useState } from 'react';

interface Props {
  onFileSelected: (file: File) => void;
}

export default function UploadZone({ onFileSelected }: Props) {
  const [dragging, setDragging] = useState(false);

  const handleFile = useCallback(
    (file: File) => {
      onFileSelected(file);
    },
    [onFileSelected],
  );

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <label
      className={`flex flex-col items-center justify-center min-h-[200px] w-full
        border-2 border-dashed rounded-xl cursor-pointer select-none transition-colors
        ${dragging ? 'border-green-400 bg-green-50' : 'border-slate-300 bg-white hover:border-slate-400 hover:bg-slate-50'}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <input
        type="file"
        accept="image/*"
        className="sr-only"
        onChange={onInputChange}
      />
      <svg
        className="w-12 h-12 text-slate-400 mb-3"
        fill="none" stroke="currentColor" viewBox="0 0 24 24"
        aria-hidden="true"
      >
        <path
          strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
          d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
        />
      </svg>
      <p className="text-slate-600 font-medium">
        Drag &amp; drop a leaf photo, or click to browse
      </p>
      <p className="text-sm text-slate-400 mt-1">JPG, PNG, WEBP up to 10 MB</p>
    </label>
  );
}
