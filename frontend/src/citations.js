export function citedNumbers(answer){
    return new Set([...answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])));
}

