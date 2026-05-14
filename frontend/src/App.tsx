import { useState } from 'react';
import { usePrediction } from './hooks/usePrediction';
import UploadZone from './components/UploadZone';
import ImagePreview from './components/ImagePreview';
import DiagnoseButton from './components/DiagnoseButton';
import ResultCard from './components/ResultCard';

const CLASS_NAMES = [
  'Apple — Apple scab', 'Apple — Black rot', 'Apple — Cedar apple rust', 'Apple — healthy',
  'Blueberry — healthy', 'Cherry — Powdery mildew', 'Cherry — healthy',
  'Corn — Cercospora leaf spot / Gray leaf spot', 'Corn — Common rust',
  'Corn — Northern Leaf Blight', 'Corn — healthy',
  'Grape — Black rot', 'Grape — Esca (Black Measles)', 'Grape — Leaf blight', 'Grape — healthy',
  'Orange — Haunglongbing (Citrus greening)',
  'Peach — Bacterial spot', 'Peach — healthy',
  'Pepper — Bacterial spot', 'Pepper — healthy',
  'Potato — Early blight', 'Potato — Late blight', 'Potato — healthy',
  'Raspberry — healthy', 'Soybean — healthy', 'Squash — Powdery mildew',
  'Strawberry — Leaf scorch', 'Strawberry — healthy',
  'Tomato — Bacterial spot', 'Tomato — Early blight', 'Tomato — Late blight',
  'Tomato — Leaf Mold', 'Tomato — Septoria leaf spot', 'Tomato — Spider mites',
  'Tomato — Target Spot', 'Tomato — Yellow Leaf Curl Virus',
  'Tomato — mosaic virus', 'Tomato — healthy',
];

function ClassList() {
  return (
    <ul className="text-sm text-slate-600 columns-2 gap-x-4 max-h-64 overflow-y-auto list-none p-0 m-0">
      {CLASS_NAMES.map((name) => (
        <li key={name} className="truncate py-0.5">{name}</li>
      ))}
    </ul>
  );
}

export default function App() {
  const { state, selectFile, reset, retrySelect, diagnose } = usePrediction();
  const [classesOpen, setClassesOpen] = useState(false);

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col items-center px-4 py-10">
      <div className="w-full max-w-lg">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-semibold text-slate-800 tracking-tight">
            Plant Disease Classifier
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Upload a leaf photo to identify disease
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 space-y-4">
          {state.status === 'idle' && (
            <>
              <UploadZone onFileSelected={selectFile} />
              <button
                type="button"
                onClick={() => setClassesOpen((o) => !o)}
                className="text-sm text-slate-500 hover:text-slate-700 underline underline-offset-2 transition-colors"
              >
                {classesOpen ? 'Hide' : 'What can this detect?'}
              </button>
              {classesOpen && <ClassList />}
            </>
          )}

          {(state.status === 'selected' || state.status === 'loading') && (
            <>
              <ImagePreview
                previewUrl={state.previewUrl}
                fileName={state.file.name}
                fileSize={state.file.size}
                onRemove={reset}
              />
              <DiagnoseButton
                loading={state.status === 'loading'}
                onClick={diagnose}
              />
            </>
          )}

          {state.status === 'result' && (
            <>
              <ImagePreview
                previewUrl={state.previewUrl}
                fileName={state.file.name}
                fileSize={state.file.size}
                onRemove={reset}
              />
              <hr className="border-slate-100" />
              <ResultCard result={state.result} onReset={reset} />
            </>
          )}

          {state.status === 'error' && (
            <>
              <ImagePreview
                previewUrl={state.previewUrl}
                fileName={state.file.name}
                fileSize={state.file.size}
                onRemove={reset}
              />
              <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3">
                <p className="text-sm text-red-800">{state.message}</p>
              </div>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={retrySelect}
                  className="flex-1 py-2 px-4 rounded-lg border border-slate-300 text-slate-600
                    hover:bg-slate-50 transition-colors text-sm font-medium"
                >
                  Try again
                </button>
                <button
                  type="button"
                  onClick={reset}
                  className="flex-1 py-2 px-4 rounded-lg border border-slate-300 text-slate-600
                    hover:bg-slate-50 transition-colors text-sm font-medium"
                >
                  Upload different image
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
