import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
	site: 'https://morning-start.github.io',
	base: '/Rag/',
	trailingSlash: 'always',

	markdown: {
		remarkPlugins: [remarkMath],
		rehypePlugins: [rehypeKatex],
	},

	integrations: [
		starlight({
			title: 'RAG 从入门到生产实践',
			description: '检索增强生成（Retrieval-Augmented Generation）全链路学习指南',
			favicon: '/favicon.svg',
			logo: {
				src: './public/favicon.svg',
				alt: 'RAG Tutorial',
			},
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/morning-start/Rag' },
			],
			editLink: {
				baseUrl: 'https://github.com/morning-start/Rag/edit/main/apps/content/',
			},
			lastUpdated: true,
			pagination: true,

			customCss: [
				'katex/dist/katex.min.css',
			],

			sidebar: [
				{ label: '首页', slug: 'index' },
				{
					label: '一、认知篇',
					collapsed: false,
					items: [
						{ label: '第1章 RAG 概述与核心价值', slug: 'chapters/ch01-rag概述与核心价值' },
						{ label: '第2章 RAG 技术架构演进（五代范式）', slug: 'chapters/ch02-rag技术架构演进' },
					],
				},
				{
					label: '二、基础篇',
					collapsed: false,
					items: [
						{ label: '第3章 环境准备与工具链', slug: 'chapters/ch03-环境准备与工具链' },
						{ label: '第4章 数据处理与索引管线', slug: 'chapters/ch04-数据处理与索引管线' },
						{ label: '第5章 检索与生成管线', slug: 'chapters/ch05-检索与生成管线' },
						{ label: '第5A章 LangGraph 完整 RAG 实战', slug: 'chapters/ch05a-langgraph完整rag实战' },
					],
				},
				{
					label: '三、进阶篇',
					collapsed: false,
					items: [
						{ label: '第6章 检索质量优化', slug: 'chapters/ch06-检索质量优化' },
						{ label: '第7章 生成质量优化', slug: 'chapters/ch07-生成质量优化' },
						{ label: '第8章 评估体系', slug: 'chapters/ch08-评估体系' },
					],
				},
				{
					label: '四、生产篇',
					collapsed: false,
					items: [
						{ label: '第9章 生产架构设计', slug: 'chapters/ch09-生产架构设计' },
						{ label: '第9A章 错误处理与高可用', slug: 'chapters/ch09a-错误处理与高可用' },
						{ label: '第10章 监控与运维', slug: 'chapters/ch10-监控与运维' },
						{ label: '第11章 安全与合规', slug: 'chapters/ch11-安全与合规' },
					],
				},
				{
					label: '五、前沿篇',
					collapsed: false,
					items: [
						{ label: '第12章 高级 RAG 范式', slug: 'chapters/ch12-高级rag范式' },
						{ label: '第12A章 多模态 RAG 展望', slug: 'chapters/ch12a-多模态rag展望' },
						{ label: '第13章 RAG + Agent 生态融合', slug: 'chapters/ch13-rag与agent生态融合' },
					],
				},
			],
		}),
	],
});
