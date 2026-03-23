/**
 * Normalizes assistant markdown so GFM tables parse reliably:
 * - Pipe-rows must be separated from preceding prose by a blank line (micromark / GFM).
 * - Collapses accidental "double separator" lines sometimes emitted by LLMs.
 */
export function prepareAssistantMarkdown(raw: string): string {
  const lines = raw.split('\n')
  const out: string[] = []

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmed = line.trimStart()
    const isPipeRow = trimmed.startsWith('|')
    const prev = out[out.length - 1]
    const prevTrim = prev?.trim() ?? ''

    if (
      isPipeRow &&
      prev !== undefined &&
      prevTrim !== '' &&
      !prevTrim.startsWith('|') &&
      !isTableSeparatorLine(prevTrim)
    ) {
      out.push('')
    }

    out.push(line)
  }

  return out.join('\n')
}

function isTableSeparatorLine(s: string): boolean {
  const t = s.trim()
  if (!t.startsWith('|')) return false
  if (!/-/.test(t)) return false
  return /^[-\s|:]+$/.test(t)
}
