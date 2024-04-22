# ZQ

```
NAME
    zq - process data with Zed queries

USAGE
    zq [ options ] [ zed-query ] file [ file ... ]

OPTIONS
    -aggmem maximum memory used per aggregate function value in MiB, MB, etc (default "auto(1GiB)")
    -B allow binary zng be sent to a terminal output (default "false")
    -C display AST in Zed canonical format (default "false")
    -color enable/disable color formatting for -Z and lake text output (default "true")
    -csv.delim CSV field delimiter (default ",")
    -e stop upon input errors (default "true")
    -f format for output data [arrows,csv,json,lake,parquet,table,text,tsv,vng,zeek,zjson,zng,zson] (default "zng")
    -fusemem maximum memory used by fuse in MiB, MB, etc (default "auto(1GiB)")
    -h display help (default "false")
    -help display help (default "false")
    -hidden show hidden options (default "false")
    -i format of input data [auto,arrows,csv,json,line,parquet,tsv,vng,zeek,zjson,zng,zson] (default "auto")
    -I source file containing Zed query text (may be used multiple times)
    -j use line-oriented JSON output independent of -f option (default "false")
    -o write data to output file
    -persist regular expression to persist type definitions across the stream
    -pretty tab size to pretty print ZSON output (0 for newline-delimited ZSON (default "4")
    -q don't display warnings (default "false")
    -s display search stats on stderr (default "false")
    -sortmem maximum memory used by sort in MiB, MB, etc (default "auto(1GiB)")
    -split split output into one file per data type in this directory (but see -splitsize)
    -splitsize if >0 and -split is set, split into files at least this big rather than by data type (default "0B")
    -unbuffered disable output buffering (default "false")
    -version print version and exit (default "false")
    -Z use formatted ZSON output independent of -f option (default "false")
    -z use line-oriented ZSON output independent of -f option (default "false")
    -zng.compress compress ZNG frames (default "true")
    -zng.framethresh minimum ZNG frame size in uncompressed bytes (default "524288")
    -zng.readmax maximum ZNG read buffer size in MiB, MB, etc. (default "auto(1GiB)")
    -zng.readsize target ZNG read buffer size in MiB, MB, etc. (default "auto(512KiB)")
    -zng.threads number of ZNG read threads (0=GOMAXPROCS) (default "0")
    -zng.validate validate format when reading ZNG (default "false")

DESCRIPTION
    "zq" is a command-line tool for processing data in diverse input formats, providing search, analytics, and extensive transormations using the Zed query language. A query
    typically applies Boolean logic or keyword search to filter the input and then transforms or analyzes the filtered stream. Output is written to one or more files or to standard
    output.

    A Zed query is comprised of one or more operators interconnected into a pipeline using the Unix pipe character "|". See https://github.com/brimdata/zed/tree/main/docs/language
    for details.

    Supported input formats include CSV, JSON, NDJSON, Parquet, VNG, ZNG, ZSON, and Zeek TSV.  Supported output formats include all the input formats along with a SQL-like table
    format.

    "zq" must be run with at least one input.  Input files can be file system paths; "-" for standard input; or HTTP, HTTPS, or S3 URLs. For most types of data, the input format is
    automatically detected. If multiple files are specified, each file format is determined independently so you can mix and match input types.  If multiple files are concatenated
    into a stream and presented as standard input, the files must all be of the same type as the beginning of stream will determine the format.

    Output is sent to standard output unless an output file is specified with -o. Some output formats like Parquet are based on schemas and require all data in the output to conform
    to the same schema.  To handle this, you can either fuse the data into a union of all the record types present (presuming all the output values are records) or you can specify
    the -split flag to indicate a destination directory for separate output files for each output type.  This flag may be used in combination with -o, which provides the prefix for
    the file path, e.g.,

    zq -f parquet -split out -o example-output input.zng

    When writing to stdout and stdout is a terminal, the default output format is ZSON. Otherwise, the default format is binary ZNG.  In either case, the default may be overridden
    with -f, -z, or -Z.

    After the options, a Zed "query" string may be specified as a single argument conforming to the Zed language syntax; i.e., it should be quoted as a single string in the shell.

    If the first argument is a path to a valid file rather than a Zed query, then the Zed query is assumed to be "*", i.e., match and output all of the input.  If the first argument
    is both a valid Zed query and an existing file, then the file overrides.

    The Zed query text may include source files using -I, which is particularly convenient when a large, complex query spans multiple lines.  In this case, these source files are
    concatenated together along with the command-line query text in the order appearing on the command line.

    The "zq" engine processes data natively in Zed so if you intend to run many queries over the same data, you will see substantial performance gains by converting your data to the
    efficient binary form of Zed called ZNG, e.g.,

    zq -f zng input.json > fast.zng   zq <query> fast.zng   ...

    Please see https://github.com/brimdata/zq and https://github.com/brimdata/zed for more information.

```