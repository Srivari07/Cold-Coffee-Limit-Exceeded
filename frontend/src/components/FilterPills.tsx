interface FilterPillsProps {
  items: string[];
  selected: string;
  onSelect: (item: string) => void;
  allLabel?: string;
}

export default function FilterPills({ items, selected, onSelect, allLabel = 'All' }: FilterPillsProps) {
  return (
    <div className="filter-pills">
      <button
        className={`pill ${selected === '' ? 'pill-active' : ''}`}
        onClick={() => onSelect('')}
      >
        {allLabel}
      </button>
      {items.map(item => (
        <button
          key={item}
          className={`pill ${selected === item ? 'pill-active' : ''}`}
          onClick={() => onSelect(item)}
        >
          {item}
        </button>
      ))}
    </div>
  );
}
