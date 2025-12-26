interface Logo1SYXProps {
  className?: string;
  size?: number;
}

export function Logo1SYX({ className = "", size = 40 }: Logo1SYXProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      <g>
        {/* Left bracket */}
        <path
          d="M 10 20 L 20 20 L 20 28 L 28 36 L 28 44 L 20 52 L 20 60 L 10 60 L 10 52 L 2 44 L 2 36 L 10 28 Z"
          fill="currentColor"
        />

        {/* Right bracket */}
        <path
          d="M 80 20 L 90 20 L 98 28 L 98 36 L 90 44 L 90 52 L 98 60 L 98 68 L 90 76 L 80 76 L 80 68 L 72 60 L 72 52 L 80 44 L 80 36 L 72 28 Z"
          fill="currentColor"
        />

        {/* Central "1" */}
        <path
          d="M 38 22 L 46 14 L 54 14 L 62 22 L 62 86 L 54 94 L 46 94 L 38 86 Z"
          fill="currentColor"
        />
      </g>
    </svg>
  );
}
