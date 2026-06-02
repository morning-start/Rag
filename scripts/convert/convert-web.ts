import * as fs from "fs";
import * as path from "path";

type FrontmatterValue = string | boolean | number | string[];

interface ParsedFrontmatter {
  [key: string]: FrontmatterValue;
}

interface ConvertOptions {
  inputDir: string;
  outputDir: string;
  dryRun?: boolean;
  verbose?: boolean;
}

interface ConvertStats {
  totalFiles: number;
  convertedFiles: number;
  skippedFiles: number;
  errors: Array<{ file: string; error: string }>;
}

const stats: ConvertStats = {
  totalFiles: 0,
  convertedFiles: 0,
  skippedFiles: 0,
  errors: [],
};

function log(message: string, level: "info" | "success" | "warning" | "error" = "info"): void {
  const icons = { info: "ℹ️", success: "✅", warning: "⚠️", error: "❌" };
  console.log(`${icons[level]} ${message}`);
}

function parseFrontmatter(content: string): {
  frontmatter: ParsedFrontmatter;
  body: string;
  rawFrontmatter: string;
} {
  const frontmatterRegex = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/;
  const match = content.match(frontmatterRegex);

  if (!match) {
    return { frontmatter: {}, body: content, rawFrontmatter: "" };
  }

  const rawFrontmatter = match[1];
  const body = content.slice(match[0].length);
  const frontmatter: ParsedFrontmatter = {};

  let currentKey = "";
  let currentValue: string[] = [];
  let inMultilineString = false;

  const processLine = (line: string) => {
    if (inMultilineString) {
      if (line.trim() === '"') {
        frontmatter[currentKey] = currentValue.join("\n").trim();
        inMultilineString = false;
        currentKey = "";
        currentValue = [];
      } else {
        currentValue.push(line);
      }
      return;
    }

    const colonIndex = line.indexOf(":");
    if (colonIndex === -1) return;

    const key = line.slice(0, colonIndex).trim();
    let value = line.slice(colonIndex + 1).trim();

    if (value.startsWith('"') && !value.endsWith('"')) {
      currentKey = key;
      inMultilineString = true;
      currentValue = [value.slice(1)];
      return;
    }

    if (value.startsWith("[")) {
      try {
        frontmatter[key] = JSON.parse(value.replace(/'/g, '"'));
      } catch {
        frontmatter[key] = value;
      }
      return;
    }

    if (value === "true") {
      frontmatter[key] = true;
    } else if (value === "false") {
      frontmatter[key] = false;
    } else if (value.startsWith('"') && value.endsWith('"')) {
      frontmatter[key] = value.slice(1, -1);
    } else if (!isNaN(Number(value)) && value !== "") {
      frontmatter[key] = Number(value);
    } else if (value !== "") {
      frontmatter[key] = value;
    }
  };

  rawFrontmatter.split("\n").forEach(processLine);

  return { frontmatter, body, rawFrontmatter };
}

function convertFilename(filename: string): string {
  const ext = path.extname(filename);
  const base = path.basename(filename, ext);

  const slug = base
    .replace(/([a-z])([A-Z])/g, "$1-$2")
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fa5-]/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");

  return `${slug}.md`;
}

function convertFrontmatter(frontmatter: ParsedFrontmatter): string {
  const lines: string[] = ["---"];

  if (frontmatter.title) {
    lines.push(`title: ${JSON.stringify(frontmatter.title)}`);
  }

  lines.push("editUrl: true");

  if (frontmatter.description) {
    lines.push(`description: ${JSON.stringify(frontmatter.description)}`);
  }

  Object.entries(frontmatter).forEach(([key, value]) => {
    if (["title", "description", "editUrl"].includes(key)) return;

    if (typeof value === "string" && value.includes("\n")) {
      lines.push(`${key}:`);
      lines.push(`  "${value}"`);
    } else if (Array.isArray(value)) {
      lines.push(`${key}: [${value.map((v) => JSON.stringify(v)).join(", ")}]`);
    } else if (typeof value === "boolean") {
      lines.push(`${key}: ${value}`);
    } else if (typeof value === "number") {
      lines.push(`${key}: ${value}`);
    } else if (typeof value === "string") {
      lines.push(`${key}: ${JSON.stringify(value)}`);
    }
  });

  lines.push("---");

  return lines.join("\n") + "\n\n";
}

function isInsideCodeBlock(line: string, inCodeBlock: boolean): boolean {
  if (line.trim().startsWith("```")) {
    return !inCodeBlock;
  }
  return inCodeBlock;
}

function convertMarkdownLinks(body: string): string {
  const lines = body.split("\n");
  let inCodeBlock = false;
  const convertedLines = lines.map((line) => {
    inCodeBlock = isInsideCodeBlock(line, inCodeBlock);

    if (inCodeBlock) return line;

    return line.replace(
      /\[([^\]]+)\]\(([^)]+\.md)(#[^)]*)?\)/g,
      (_match, text, filepath, anchor) => {
        const newFilepath = filepath
          .replace(/\.md$/, "/")
          .replace(/^\.\/reference\//, "../reference/");
        return `[${text}](${newFilepath}${anchor || ""})`;
      },
    );
  });

  return convertedLines.join("\n");
}

function convertImagePaths(body: string): string {
  return body.replace(
    /!\[([^\]]*)\]\(([^)]+\.(png|jpg|jpeg|gif|webp|svg))\)/gi,
    (_match, alt, src) => `![${alt}](${src})`,
  );
}

function normalizeWhitespace(body: string): string {
  return body
    .replace(/\r\n/g, "\n")
    .replace(/\n{4,}/g, "\n\n\n")
    .replace(/[ \t]+$/gm, "")
    .trimEnd() + "\n";
}

function convertBody(body: string): string {
  let converted = body;

  converted = convertMarkdownLinks(converted);
  converted = convertImagePaths(converted);
  converted = normalizeWhitespace(converted);

  return converted;
}

function validateContent(content: string, filePath: string): boolean {
  if (!content.trim()) {
    log(`Empty file: ${filePath}`, "warning");
    return false;
  }

  if (!content.includes("#")) {
    log(`No heading found: ${filePath}`, "warning");
  }

  return true;
}

async function convertFile(
  inputPath: string,
  outputPath: string,
  options: ConvertOptions,
): Promise<void> {
  try {
    const content = fs.readFileSync(inputPath, "utf-8");

    if (!validateContent(content, inputPath)) {
      stats.skippedFiles++;
      return;
    }

    const { frontmatter, body } = parseFrontmatter(content);

    if (options.verbose) {
      log(`Parsing frontmatter for: ${path.basename(inputPath)}`, "info");
      log(`  Keys found: ${Object.keys(frontmatter).join(", ")}`, "info");
    }

    const newFrontmatter = convertFrontmatter(frontmatter);
    const newBody = convertBody(body);
    const result = newFrontmatter + newBody;

    if (options.dryRun) {
      log(`[DRY RUN] Would convert: ${inputPath} -> ${outputPath}`);
      log(`[DRY RUN] Output length: ${result.length} chars`, "info");
      stats.convertedFiles++;
      return;
    }

    const outputDir = path.dirname(outputPath);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    fs.writeFileSync(outputPath, result, "utf-8");
    stats.convertedFiles++;
    log(`Converted: ${path.basename(inputPath)} -> ${path.relative(process.cwd(), outputPath)}`, "success");
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error);
    stats.errors.push({ file: inputPath, error: errorMsg });
    log(`Error converting ${path.basename(inputPath)}: ${errorMsg}`, "error");
  }
}

export async function convertWeb(options: Partial<ConvertOptions> = {}): Promise<void> {
  Object.assign(stats, {
    totalFiles: 0,
    convertedFiles: 0,
    skippedFiles: 0,
    errors: [],
  });

  const defaultOptions: ConvertOptions = {
    inputDir: path.resolve(process.cwd(), "apps/content"),
    outputDir: path.resolve(process.cwd(), "apps/web/src/convert"),
    dryRun: false,
    verbose: false,
    ...options,
  };

  if (!fs.existsSync(defaultOptions.inputDir)) {
    log(`Input directory not found: ${defaultOptions.inputDir}`, "error");
    process.exit(1);
  }

  log(`📝 Converting content from: ${defaultOptions.inputDir}`);
  log(`📂 Output directory: ${defaultOptions.outputDir}`);
  log("");

  const files = fs.readdirSync(defaultOptions.inputDir).filter((f) => f.endsWith(".md"));

  if (files.length === 0) {
    log("No markdown files found.", "warning");
    return;
  }

  stats.totalFiles = files.length;
  log(`Found ${files.length} files to convert.\n`);

  const conversionPromises = files.map((file) => {
    const inputPath = path.join(defaultOptions.inputDir, file);
    const outputFilename = convertFilename(file);
    const outputPath = path.join(defaultOptions.outputDir, outputFilename);

    return convertFile(inputPath, outputPath, defaultOptions);
  });

  await Promise.all(conversionPromises);

  log("");
  log("=== Conversion Summary ===", "info");
  log(`Total files: ${stats.totalFiles}`, "info");
  log(`Converted: ${stats.convertedFiles}`, "success");
  log(`Skipped: ${stats.skippedFiles}`, "warning");

  if (stats.errors.length > 0) {
    log(`Errors: ${stats.errors.length}`, "error");
    stats.errors.forEach((err) => log(`  - ${err.file}: ${err.error}`, "error"));
  } else {
    log("✨ All conversions completed successfully!", "success");
  }
}

const isMainModule =
  import.meta.url === `file://${process.argv[1]?.replace(/\\/g, "/")}` || Bun.main === import.meta.path;

if (isMainModule) {
  const args = process.argv.slice(2);
  const dryRun = args.includes("--dry-run") || args.includes("-n");
  const verbose = args.includes("--verbose") || args.includes("-v");

  convertWeb({ dryRun, verbose }).catch((err) => {
    log(`Fatal error: ${err.message}`, "error");
    process.exit(1);
  });
}
