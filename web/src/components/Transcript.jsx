import { useEffect, useRef, useState } from 'react';

/**
 * The transcript surface.
 *
 * Not a contenteditable. Each word is its own element carrying the id that ties
 * it to a span of samples, and edits are expressed as operations on those
 * blocks. A contenteditable would hand us a flat string on every keystroke and
 * we would have to re-derive which words survived — exactly the information we
 * already have and must not throw away.
 *
 * Deleted words stay on screen, struck through, because in a transcript editor
 * "what did I cut?" is asked constantly and a deletion that vanishes is a
 * deletion you cannot review.
 */
export default function Transcript({
  doc,
  selection,
  caret,
  activeIndex,
  showDeleted,
  onSelect,
  onCaret,
  onSeek,
  onCommitInsert,
  onUpdateInsert,
  draft,
  onDraftChange,
}) {
  const containerRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const inSelection = (index) =>
    selection && index >= Math.min(selection.anchor, selection.focus)
      && index <= Math.max(selection.anchor, selection.focus);

  // Keep the word under the playhead in view during playback.
  useEffect(() => {
    if (activeIndex < 0 || !containerRef.current) return;
    const node = containerRef.current.querySelector(`[data-block="${activeIndex}"]`);
    if (!node) return;
    const box = node.getBoundingClientRect();
    const frame = containerRef.current.getBoundingClientRect();
    if (box.top < frame.top + 40 || box.bottom > frame.bottom - 40) {
      node.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }
  }, [activeIndex]);

  const paragraphs = groupIntoParagraphs(doc);

  return (
    <div className="transcript" ref={containerRef} onMouseUp={() => setDragging(false)}>
      {paragraphs.map((paragraph, pIndex) => (
        <p key={pIndex} className="paragraph">
          {paragraph.map(({ block, index }) => {
            const gapBefore = (
              <Gap
                key={`gap-${index}`}
                position={index}
                active={caret === index}
                draft={draft?.position === index ? draft : null}
                onClick={() => onCaret(index)}
                onDraftChange={onDraftChange}
                onCommit={onCommitInsert}
              />
            );

            if (block.kind === 'insert') {
              return (
                <span key={block.id} className="block">
                  {gapBefore}
                  <InsertChip
                    block={block}
                    selected={inSelection(index)}
                    onChange={(text) => onUpdateInsert(block.id, text)}
                    onMouseDown={(event) => {
                      event.preventDefault();
                      onSelect({ anchor: index, focus: index });
                      setDragging(true);
                    }}
                  />
                </span>
              );
            }

            if (block.deleted && !showDeleted) return gapBefore;

            return (
              <span key={block.id} className="block">
                {gapBefore}
                <span
                  data-block={index}
                  className={[
                    'word',
                    block.deleted ? 'is-deleted' : '',
                    inSelection(index) ? 'is-selected' : '',
                    activeIndex === index ? 'is-playing' : '',
                  ].join(' ')}
                  title={`${block.start.toFixed(2)}s – ${block.end.toFixed(2)}s${
                    block.deleted ? ' (cut — click to restore)' : ''
                  }`}
                  onMouseDown={(event) => {
                    event.preventDefault();
                    if (event.shiftKey && selection) onSelect({ ...selection, focus: index });
                    else {
                      onSelect({ anchor: index, focus: index });
                      setDragging(true);
                    }
                    onCaret(null);
                  }}
                  onMouseEnter={() => {
                    if (dragging && selection) onSelect({ ...selection, focus: index });
                  }}
                  onDoubleClick={() => onSeek(block.start)}
                >
                  {block.text}
                </span>
              </span>
            );
          })}
          {pIndex === paragraphs.length - 1 && (
            <Gap
              position={doc.length}
              active={caret === doc.length}
              draft={draft?.position === doc.length ? draft : null}
              onClick={() => onCaret(doc.length)}
              onDraftChange={onDraftChange}
              onCommit={onCommitInsert}
              trailing
            />
          )}
        </p>
      ))}
      {doc.length === 0 && <p className="empty">This recording produced no transcript.</p>}
    </div>
  );
}

/** The clickable space between two words, where new text goes. */
function Gap({ position, active, draft, onClick, onDraftChange, onCommit, trailing }) {
  const inputRef = useRef(null);

  useEffect(() => {
    if (draft && inputRef.current) inputRef.current.focus();
  }, [draft]);

  if (draft) {
    return (
      <span className="draft">
        <input
          ref={inputRef}
          className="draft-input"
          value={draft.text}
          placeholder="type words to synthesise…"
          size={Math.max(draft.text.length + 1, 12)}
          onChange={(event) => onDraftChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault();
              onCommit(true);
            } else if (event.key === 'Escape') {
              event.preventDefault();
              onCommit(false);
            }
            event.stopPropagation();
          }}
          onBlur={() => onCommit(true)}
        />
      </span>
    );
  }

  return (
    <span
      className={`gap ${active ? 'is-active' : ''} ${trailing ? 'is-trailing' : ''}`}
      onMouseDown={(event) => {
        event.preventDefault();
        onClick(position);
      }}
      title="Click here, then type, to add words"
    />
  );
}

/** Inserted text, visually distinct because it will be synthesised. */
function InsertChip({ block, selected, onChange, onMouseDown }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(block.text);

  useEffect(() => setValue(block.text), [block.text]);

  if (editing) {
    return (
      <input
        className="draft-input"
        autoFocus
        value={value}
        size={Math.max(value.length + 1, 8)}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === 'Escape') {
            event.preventDefault();
            setEditing(false);
            onChange(event.key === 'Enter' ? value : block.text);
          }
          event.stopPropagation();
        }}
        onBlur={() => {
          setEditing(false);
          onChange(value);
        }}
      />
    );
  }

  return (
    <span
      className={`word is-inserted ${selected ? 'is-selected' : ''}`}
      title="Added text — this will be synthesised. Double-click to edit."
      onMouseDown={onMouseDown}
      onDoubleClick={() => setEditing(true)}
    >
      {block.text}
    </span>
  );
}

/** Split blocks into paragraphs on the ASR's segment boundaries. */
function groupIntoParagraphs(doc) {
  const paragraphs = [];
  let current = [];
  let segment = null;

  doc.forEach((block, index) => {
    if (block.kind === 'word' && segment !== null && block.segment !== segment && current.length) {
      paragraphs.push(current);
      current = [];
    }
    if (block.kind === 'word') segment = block.segment;
    current.push({ block, index });
  });
  if (current.length) paragraphs.push(current);
  return paragraphs.length ? paragraphs : [[]];
}
