export default function Passages({ passages, cited }) {
  return (
    <div className="mt-8">
      <h2 className="text-xs font-medium uppercase tracking-wide text-stone-500">
        Retrieved passages
      </h2>
      <ol className="mt-2 space-y-3">
        {passages.map((p) => {
          const isCited = cited.has(p.number);
          return (
            <li
              key={p.number}
              id={`passage-${p.number}`}
              className={`scroll-mt-4 rounded-md border px-4 py-3 ${
                isCited
                  ? "border-orange-400 bg-orange-50"
                  : "border-stone-200 bg-white"
              }`}
            >
              <div className="flex items-center gap-2 text-xs text-stone-500">
                <span className="font-mono font-semibold text-stone-800">
                  [{p.number}]
                </span>
                {isCited && (
                  <span className="rounded bg-orange-200 px-1.5 py-0.5 font-medium text-orange-900">
                    cited
                  </span>
                )}
                <span className="ml-auto font-mono">
                  score {p.score.toFixed(3)} · chunk {p.chunk_index}
                </span>
              </div>
              <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-stone-700">
                {p.text}
              </p>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
