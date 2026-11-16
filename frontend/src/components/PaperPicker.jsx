import { useState } from "react";

export default function PaperPicker({
  papers,
  paperId,
  onPaperChange,
  questions,
  onPickQuestion,
}) {
  const [filter, setFilter] = useState("");
  const f = filter.trim().toLowerCase();

  // Keep the selected paper in the list even if it doesn't match the filter.
  const shown = papers.filter(
    (p) =>
      p.id === paperId || p.title.toLowerCase().includes(f) || p.id.includes(f),
  );

  return (
    <aside className="space-y-6">
      <div>
        <label className="text-xs font-medium uppercase tracking-wide text-stone-500">
          Paper
        </label>
        <input
          type="text"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter by title or id…"
          className="mt-2 w-full rounded-md border border-stone-300 bg-white px-3 py-2 text-sm outline-none focus:border-orange-700"
        />
        <select
          value={paperId}
          onChange={(e) => onPaperChange(e.target.value)}
          className="mt-2 w-full rounded-md border border-stone-300 bg-white px-3 py-2 text-sm outline-none focus:border-orange-700"
        >
          {shown.map((p) => (
            <option key={p.id} value={p.id}>
              {p.title}
            </option>
          ))}
        </select>
        <p className="mt-1 text-xs text-stone-400">
          {shown.length} of {papers.length} papers
        </p>
      </div>

      <div>
        <h2 className="text-xs font-medium uppercase tracking-wide text-stone-500">
          Questions from QASPER
        </h2>
        {questions.length === 0 ? (
          <p className="mt-2 text-sm text-stone-400">
            No suggested questions for this paper.
          </p>
        ) : (
          <ul className="mt-2 space-y-1">
            {questions.map((q) => (
              <li key={q}>
                <button
                  type="button"
                  onClick={() => onPickQuestion(q)}
                  className="w-full rounded-md px-2 py-1.5 text-left text-sm text-stone-700 hover:bg-stone-200/60"
                >
                  {q}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  );
}
