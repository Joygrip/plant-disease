import LoadingSpinner from './LoadingSpinner';

interface Props {
  loading: boolean;
  onClick: () => void;
}

export default function DiagnoseButton({ loading, onClick }: Props) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={loading}
      className="flex items-center justify-center gap-2 w-full py-3 px-6 rounded-lg
        bg-green-600 hover:bg-green-700 disabled:opacity-70 disabled:cursor-not-allowed
        text-white font-semibold transition-colors"
    >
      {loading ? (
        <>
          <LoadingSpinner />
          Analysing…
        </>
      ) : (
        'Diagnose'
      )}
    </button>
  );
}
