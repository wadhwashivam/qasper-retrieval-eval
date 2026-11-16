export default function Answer({ answer }) {
  // Split while keeping the [n] markers; odd-indexed parts are citations.
    const parts = answer.split(/(\[\d+\])/);

    return (
        <div className="mt-8">
        <h2 className="text-xs font-medium uppercase tracking-wide text-stone-500">Answer</h2>
        <p className="mt-2 rounded-md border border-stone-200 bg-white px-4 py-3 leading-relaxed">
            {parts.map((part, i) => {
            if (i % 2 === 0) return <span key={i}>{part}</span>;
            const n = part.slice(1, -1);
            return (
                <a
                key={i}
                href={`#passage-${n}`}
                className="mx-0.5 rounded bg-orange-100 px-1 font-mono text-xs text-orange-800 hover:bg-orange-200"
                >
                {n}
                </a>
            );
            })}
        </p>
        </div>
    );
}