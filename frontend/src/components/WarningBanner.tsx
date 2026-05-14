interface Props {
  message: string;
}

export default function WarningBanner({ message }: Props) {
  return (
    <div className="rounded-lg border border-yellow-300 bg-yellow-50 px-4 py-3">
      <p className="text-sm text-yellow-800 flex items-start gap-2">
        <span aria-hidden="true">⚠</span>
        {message}
      </p>
    </div>
  );
}
