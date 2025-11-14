#!/usr/bin/env python
"""
Command-line script for preprocessing ROOT files.

This script replaces the C++ preprocessing functionality with a pure Python
implementation. It can output to either ROOT or NPZ format.

Usage:
    python preprocess_root.py input.root output.root --tree reco
    python preprocess_root.py input.root output.npz --tree reco --format npz --nu-flows
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.RootPreprocessor import preprocess_root_file, preprocess_root_directory


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ROOT files for ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess single file to ROOT format
  %(prog)s input.root output.root --tree reco
  
  # Preprocess single file to NPZ format with NuFlow results
  %(prog)s input.root output.npz --format npz --nu-flows
  
  # Preprocess directory and merge all files
  %(prog)s input_dir/ output.npz --format npz --merge
  
  # Preprocess directory, keep files separate
  %(prog)s input_dir/ output_dir/ --format npz --no-merge
  
  # Include initial parton information
  %(prog)s input.root output.root --tree reco --initial-parton-info
  
  # Process with custom file pattern
  %(prog)s input_dir/ output.npz --format npz --pattern "data_*.root"
  
  # Process quietly
  %(prog)s input.root output.npz --format npz --quiet
        """
    )
    
    parser.add_argument(
        "input",
        help="Input ROOT file path or directory containing ROOT files"
    )
    parser.add_argument(
        "output",
        help="Output file path (.root or .npz) or output directory (if --no-merge)"
    )
    parser.add_argument(
        "--tree",
        default="reco",
        help="Name of TTree in input file (default: reco)"
    )
    parser.add_argument(
        "--format",
        choices=["root", "npz"],
        default=None,
        help="Output format (default: inferred from output file extension)"
    )
    parser.add_argument(
        "--nu-flows",
        action="store_true",
        help="Include NuFlow neutrino reconstruction results"
    )
    parser.add_argument(
        "--initial-parton-info",
        action="store_true",
        help="Include initial state parton information"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        default=None,
        help="Merge all files in directory into one output (default for file output)"
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Keep files separate when processing directory (default for directory output)"
    )
    parser.add_argument(
        "--pattern",
        default="*.root",
        help="Glob pattern for ROOT files in directory (default: *.root)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input path does not exist: {args.input}", file=sys.stderr)
        return 1
    
    is_directory = os.path.isdir(args.input)
    
    # Determine merge behavior
    if args.merge and args.no_merge:
        print("Error: Cannot specify both --merge and --no-merge", file=sys.stderr)
        return 1
    
    if is_directory:
        # For directories, infer merge from output type
        if args.merge is None and args.no_merge is None:
            # Default: merge if output looks like a file, don't merge if it looks like a dir
            output_has_ext = os.path.splitext(args.output)[1] in ['.root', '.npz', '.ROOT', '.NPZ']
            merge = output_has_ext
        else:
            merge = args.merge if args.merge is not None else not args.no_merge
    else:
        merge = True  # Not used for single files
    
    # Infer output format from extension if not specified
    if args.format is None:
        if is_directory and not merge:
            # For non-merged directory output, check if output path suggests format
            if args.output.endswith('_npz') or 'npz' in args.output.lower():
                args.format = "npz"
            elif args.output.endswith('_root') or 'root' in args.output.lower():
                args.format = "root"
            else:
                # Default to npz for directory processing
                args.format = "npz"
        else:
            ext = os.path.splitext(args.output)[1].lower()
            if ext == ".npz":
                args.format = "npz"
            elif ext == ".root":
                args.format = "root"
            else:
                print(
                    f"Error: Cannot infer output format from extension '{ext}'. "
                    "Please specify --format",
                    file=sys.stderr
                )
                return 1
    
    # Warn if output exists
    if os.path.exists(args.output) and not args.quiet:
        if os.path.isfile(args.output):
            print(f"Warning: Output file already exists and will be overwritten: {args.output}")
        else:
            print(f"Warning: Output directory already exists: {args.output}")
    
    # Run preprocessing
    try:
        if is_directory:
            preprocess_root_directory(
                input_dir=args.input,
                output_path=args.output,
                tree_name=args.tree,
                output_format=args.format,
                save_nu_flows=args.nu_flows,
                save_initial_parton_info=args.initial_parton_info,
                merge=merge,
                pattern=args.pattern,
                verbose=not args.quiet,
            )
        else:
            preprocess_root_file(
                input_path=args.input,
                output_path=args.output,
                tree_name=args.tree,
                output_format=args.format,
                save_nu_flows=args.nu_flows,
                save_initial_parton_info=args.initial_parton_info,
                verbose=not args.quiet,
            )
        
        if not args.quiet:
            print("\nPreprocessing completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"Error during preprocessing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
