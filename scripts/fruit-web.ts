import * as fs from 'fs';
import * as path from 'path';

interface FruitConfig {
	convertDir: string;
	outputDir: string;
	dryRun?: boolean;
}

interface FruitContent {
	before?: string;
	after?: string;
	[key: string]: string | undefined;
}

function loadFruitConfig(filePath: string): FruitContent | null {
	if (!fs.existsSync(filePath)) {
		return null;
	}

	const content = fs.readFileSync(filePath, 'utf-8');
	const config: FruitContent = {};

	const sections = content.split(/^### (.+)$/m);
	for (let i = 1; i < sections.length; i += 2) {
		const key = sections[i]?.trim();
		const value = sections[i + 1]?.trim();
		if (key && value) {
			config[key] = value;
		}
	}

	return Object.keys(config).length > 0 ? config : null;
}

function processFile(inputPath: string, outputPath: string, fruitDir: string, options: FruitConfig): void {
	const baseName = path.basename(inputPath, '.md');
	const fruitPath = path.join(fruitDir, `${baseName}.fruit.md`);

	const content = fs.readFileSync(inputPath, 'utf-8');
	const fruitConfig = loadFruitConfig(fruitPath);

	let result = content;

	if (fruitConfig) {
		if (fruitConfig.before) {
			result = fruitConfig.before + '\n\n' + result;
		}

		if (fruitConfig.after) {
			result = result + '\n\n' + fruitConfig.after;
		}

		console.log(`  🍒 Applied fruit content for: ${baseName}`);
	} else {
		console.log(`  📄 No fruit content found for: ${baseName}`);
	}

	if (options.dryRun) {
		console.log(`[DRY RUN] Would output to: ${outputPath}`);
		return;
	}

	const outputDir = path.dirname(outputPath);
	if (!fs.existsSync(outputDir)) {
		fs.mkdirSync(outputDir, { recursive: true });
	}

	fs.writeFileSync(outputPath, result, 'utf-8');
}

export function fruitWeb(options: Partial<FruitConfig> = {}): void {
	const defaultOptions: FruitConfig = {
		convertDir: path.resolve(process.cwd(), 'apps/web/src/convert'),
		outputDir: path.resolve(process.cwd(), 'apps/web/src/fruit'),
		dryRun: false,
		...options,
	};

	const fruitDir = path.resolve(process.cwd(), 'scripts/fruit');

	if (!fs.existsSync(defaultOptions.convertDir)) {
		console.error(`Convert directory not found: ${defaultOptions.convertDir}`);
		console.error('Please run convert-web first.');
		process.exit(1);
	}

	console.log(`🍒 Processing fruit content`);
	console.log(`   Input: ${defaultOptions.convertDir}`);
	console.log(`   Output: ${defaultOptions.outputDir}`);
	console.log(`   Fruit configs: ${fruitDir}`);
	console.log('');

	const files = fs.readdirSync(defaultOptions.convertDir).filter((f) => f.endsWith('.md'));

	if (files.length === 0) {
		console.log('No converted files found. Run convert-web first.');
		return;
	}

	console.log(`Found ${files.length} files to process.\n`);

	for (const file of files) {
		const inputPath = path.join(defaultOptions.convertDir, file);
		const outputPath = path.join(defaultOptions.outputDir, file);

		processFile(inputPath, outputPath, fruitDir, defaultOptions);
	}

	console.log(`\n✅ Fruit processing complete! ${files.length} files processed.`);
}

function initFruitTemplate(filename: string): void {
	const fruitDir = path.resolve(process.cwd(), 'scripts/fruit');
	const templatePath = path.join(fruitDir, `${filename}.fruit.md`);

	if (fs.existsSync(templatePath)) {
		console.log(`Fruit template already exists: ${templatePath}`);
		return;
	}

	if (!fs.existsSync(fruitDir)) {
		fs.mkdirSync(fruitDir, { recursive: true });
	}

	const template = `# Fruit Content for ${filename}

## Content added before the main content (optional)

### before

<!-- Add content that should appear BEFORE the main content here -->
<!-- Example: {: .note } This chapter covers advanced topics... -->


## Content added after the main content (optional)

### after

<!-- Add content that should appear AFTER the main content here -->
<!-- Example: {: .tip } Want to learn more? Check out [next chapter](...) -->
`;

	fs.writeFileSync(templatePath, template, 'utf-8');
	console.log(`Created fruit template: ${templatePath}`);
}

const isMainModule = import.meta.url === `file://${process.argv[1]?.replace(/\\/g, '/')}` || Bun.main === import.meta.path;

if (isMainModule) {
	const args = process.argv.slice(2);
	const dryRun = args.includes('--dry-run') || args.includes('-n');

	if (args.includes('init')) {
		const filename = args[1];
		if (!filename) {
			console.error('Usage: fruit-web init <filename>');
			process.exit(1);
		}
		initFruitTemplate(filename);
	} else {
		fruitWeb({ dryRun });
	}
}
