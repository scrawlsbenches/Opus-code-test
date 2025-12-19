/* ============================================================================
   The Cortical Chronicles - Application
   Navigation, search, and markdown rendering
   ============================================================================ */

(function() {
    'use strict';

    // State
    let indexData = null;
    let currentChapter = null;

    // DOM Elements
    const navTree = document.getElementById('nav-tree');
    const chapterContent = document.getElementById('chapter');
    const searchInput = document.getElementById('search');

    // --------------------------------------------------------------------------
    // Initialization
    // --------------------------------------------------------------------------

    async function init() {
        try {
            // Load index data
            const response = await fetch('index.json');
            if (!response.ok) {
                throw new Error('index.json not found - another agent may still be generating it');
            }
            indexData = await response.json();

            // Build navigation
            buildNavigation();

            // Setup search
            searchInput.addEventListener('input', handleSearch);

            // Handle hash navigation
            window.addEventListener('hashchange', handleHashChange);

            // Load initial chapter
            handleHashChange();

        } catch (error) {
            showError('Failed to initialize: ' + error.message);
        }
    }

    // --------------------------------------------------------------------------
    // Navigation
    // --------------------------------------------------------------------------

    function buildNavigation() {
        if (!indexData || !indexData.chapters) {
            navTree.innerHTML = '<p class="error">No content available</p>';
            return;
        }

        navTree.innerHTML = '';

        // Group chapters by section
        const sections = {};
        indexData.chapters.forEach(chapter => {
            const section = chapter.section || 'other';
            if (!sections[section]) {
                sections[section] = [];
            }
            sections[section].push(chapter);
        });

        // Section display order and titles
        const sectionOrder = ['preface', 'foundations', 'architecture', 'decisions', 'evolution', 'future'];
        const sectionTitles = {
            'preface': 'Preface',
            'foundations': 'Foundations',
            'architecture': 'Architecture',
            'decisions': 'Decisions',
            'evolution': 'Evolution',
            'future': 'Future'
        };

        // Build navigation tree
        sectionOrder.forEach(sectionKey => {
            if (!sections[sectionKey]) return;

            const sectionDiv = document.createElement('div');
            sectionDiv.className = 'nav-section';

            const sectionTitle = document.createElement('div');
            sectionTitle.className = 'nav-section-title';
            sectionTitle.textContent = sectionTitles[sectionKey] || sectionKey;
            sectionDiv.appendChild(sectionTitle);

            sections[sectionKey].forEach(chapter => {
                const chapterDiv = document.createElement('div');
                chapterDiv.className = 'nav-chapter';
                chapterDiv.textContent = chapter.title;
                chapterDiv.dataset.chapterPath = chapter.path;
                chapterDiv.dataset.title = chapter.title.toLowerCase();
                chapterDiv.addEventListener('click', () => navigateToChapter(chapter.path));
                sectionDiv.appendChild(chapterDiv);
            });

            navTree.appendChild(sectionDiv);
        });
    }

    function navigateToChapter(chapterPath) {
        // Remove .md extension for the hash
        const hashPath = chapterPath.replace(/\.md$/, '');
        window.location.hash = hashPath;
    }

    async function handleHashChange() {
        const hash = window.location.hash.slice(1); // Remove #

        if (!hash) {
            showWelcome();
            updateActiveNav(null);
            return;
        }

        // Update active state in navigation
        updateActiveNav(hash);

        // Load chapter
        await loadChapter(hash);
    }

    function updateActiveNav(chapterHash) {
        const allChapters = navTree.querySelectorAll('.nav-chapter');
        allChapters.forEach(chapter => {
            const chapterHash2 = chapter.dataset.chapterPath.replace(/\.md$/, '');
            if (chapterHash2 === chapterHash) {
                chapter.classList.add('active');
            } else {
                chapter.classList.remove('active');
            }
        });
    }

    // --------------------------------------------------------------------------
    // Chapter Loading
    // --------------------------------------------------------------------------

    async function loadChapter(chapterHash) {
        try {
            showLoading();

            // Add .md extension if not present
            const chapterPath = chapterHash.endsWith('.md') ? chapterHash : `${chapterHash}.md`;

            const response = await fetch(chapterPath);
            if (!response.ok) {
                throw new Error(`Chapter not found: ${chapterPath}`);
            }

            const markdown = await response.text();
            const html = renderMarkdown(markdown);

            chapterContent.innerHTML = html;
            currentChapter = chapterHash;

            // Scroll to top
            document.getElementById('content').scrollTop = 0;

        } catch (error) {
            showError('Failed to load chapter: ' + error.message);
        }
    }

    function showWelcome() {
        chapterContent.innerHTML = `
            <h1>📚 The Cortical Chronicles</h1>
            <p>Welcome to the self-documenting book about the Cortical Text Processor.</p>
            <p>Select a chapter from the navigation to begin reading.</p>
            <h2>About This Book</h2>
            <p>This book is automatically generated from the project's source code, commit history, and architectural decisions. It provides:</p>
            <ul>
                <li><strong>Algorithm Documentation</strong> - Deep dives into PageRank, BM25, and semantic extraction</li>
                <li><strong>Architecture Guides</strong> - Module-by-module explanations</li>
                <li><strong>Evolution Timeline</strong> - How the project grew and changed</li>
                <li><strong>Decision Records</strong> - Why certain choices were made</li>
            </ul>
        `;
    }

    function showLoading() {
        chapterContent.innerHTML = '<div class="loading">Loading chapter...</div>';
    }

    function showError(message) {
        chapterContent.innerHTML = `<div class="error">${escapeHtml(message)}</div>`;
    }

    // --------------------------------------------------------------------------
    // Search
    // --------------------------------------------------------------------------

    function handleSearch(e) {
        const query = e.target.value.toLowerCase().trim();
        const chapters = navTree.querySelectorAll('.nav-chapter');

        if (!query) {
            // Show all chapters
            chapters.forEach(chapter => chapter.classList.remove('hidden'));
            return;
        }

        // Filter chapters by title
        chapters.forEach(chapter => {
            const title = chapter.dataset.title || '';
            if (title.includes(query)) {
                chapter.classList.remove('hidden');
            } else {
                chapter.classList.add('hidden');
            }
        });
    }

    // --------------------------------------------------------------------------
    // Markdown Rendering
    // --------------------------------------------------------------------------

    function renderMarkdown(markdown) {
        let html = markdown;

        // Strip YAML frontmatter if present
        if (html.startsWith('---')) {
            const endIndex = html.indexOf('---', 3);
            if (endIndex !== -1) {
                html = html.substring(endIndex + 3).trim();
            }
        }

        // Escape HTML first (will be unescaped in specific contexts)
        html = escapeHtml(html);

        // Code blocks (must be before inline code)
        html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
            const highlighted = highlightCode(code, lang);
            return `<pre><code class="language-${lang || 'text'}">${highlighted}</code></pre>`;
        });

        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Headers
        html = html.replace(/^##### (.*$)/gim, '<h5>$1</h5>');
        html = html.replace(/^#### (.*$)/gim, '<h4>$1</h4>');
        html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
        html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
        html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

        // Bold and italic
        html = html.replace(/\*\*\*([^*]+)\*\*\*/g, '<strong><em>$1</em></strong>');
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
        html = html.replace(/___([^_]+)___/g, '<strong><em>$1</em></strong>');
        html = html.replace(/__([^_]+)__/g, '<strong>$1</strong>');
        html = html.replace(/_([^_]+)_/g, '<em>$1</em>');

        // Links [text](url)
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');

        // Horizontal rules
        html = html.replace(/^---$/gim, '<hr>');
        html = html.replace(/^\*\*\*$/gim, '<hr>');

        // Blockquotes
        html = html.replace(/^&gt; (.*)$/gim, '<blockquote>$1</blockquote>');

        // Lists - ordered
        html = html.replace(/^\d+\.\s+(.*)$/gim, '<li>$1</li>');

        // Lists - unordered
        html = html.replace(/^[-*]\s+(.*)$/gim, '<li>$1</li>');

        // Wrap consecutive list items
        html = html.replace(/(<li>.*<\/li>\n?)+/g, (match) => {
            // Check if previous line was ordered list
            if (/^\d+\./.test(match)) {
                return '<ol>' + match + '</ol>';
            }
            return '<ul>' + match + '</ul>';
        });

        // Tables
        html = renderTables(html);

        // Paragraphs (lines separated by blank lines)
        const lines = html.split('\n');
        const paragraphs = [];
        let currentParagraph = [];

        lines.forEach(line => {
            const trimmed = line.trim();

            // Skip if it's already a block element
            if (trimmed.startsWith('<h') ||
                trimmed.startsWith('<pre>') ||
                trimmed.startsWith('<ul>') ||
                trimmed.startsWith('<ol>') ||
                trimmed.startsWith('<blockquote>') ||
                trimmed.startsWith('<hr>') ||
                trimmed === '') {

                // Flush current paragraph
                if (currentParagraph.length > 0) {
                    paragraphs.push('<p>' + currentParagraph.join(' ') + '</p>');
                    currentParagraph = [];
                }

                if (trimmed !== '') {
                    paragraphs.push(line);
                }
            } else {
                currentParagraph.push(trimmed);
            }
        });

        // Flush remaining paragraph
        if (currentParagraph.length > 0) {
            paragraphs.push('<p>' + currentParagraph.join(' ') + '</p>');
        }

        html = paragraphs.join('\n');

        return html;
    }

    // --------------------------------------------------------------------------
    // Table Rendering
    // --------------------------------------------------------------------------

    function renderTables(html) {
        const lines = html.split('\n');
        const result = [];
        let inTable = false;
        let tableRows = [];

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();

            // Check if line is a table row
            if (line.startsWith('|') && line.endsWith('|')) {
                if (!inTable) {
                    inTable = true;
                    tableRows = [];
                }
                tableRows.push(line);
            } else {
                // End of table
                if (inTable && tableRows.length > 0) {
                    result.push(buildTable(tableRows));
                    tableRows = [];
                    inTable = false;
                }
                result.push(lines[i]);
            }
        }

        // Handle table at end of file
        if (inTable && tableRows.length > 0) {
            result.push(buildTable(tableRows));
        }

        return result.join('\n');
    }

    function buildTable(rows) {
        if (rows.length < 2) return rows.join('\n');

        let html = '<table>\n';

        // First row is header
        const headerCells = rows[0].split('|').slice(1, -1).map(cell => cell.trim());
        html += '<thead>\n<tr>\n';
        headerCells.forEach(cell => {
            html += `<th>${cell}</th>\n`;
        });
        html += '</tr>\n</thead>\n';

        // Second row is separator (skip)
        // Remaining rows are body
        if (rows.length > 2) {
            html += '<tbody>\n';
            for (let i = 2; i < rows.length; i++) {
                const cells = rows[i].split('|').slice(1, -1).map(cell => cell.trim());
                html += '<tr>\n';
                cells.forEach(cell => {
                    html += `<td>${cell}</td>\n`;
                });
                html += '</tr>\n';
            }
            html += '</tbody>\n';
        }

        html += '</table>';
        return html;
    }

    // --------------------------------------------------------------------------
    // Syntax Highlighting (Simple)
    // --------------------------------------------------------------------------

    function highlightCode(code, lang) {
        // Simple keyword highlighting for common languages
        const keywords = {
            python: ['def', 'class', 'import', 'from', 'return', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'with', 'as', 'pass', 'break', 'continue', 'yield', 'async', 'await', 'lambda', 'None', 'True', 'False'],
            javascript: ['function', 'const', 'let', 'var', 'return', 'if', 'else', 'for', 'while', 'try', 'catch', 'finally', 'class', 'extends', 'import', 'export', 'default', 'async', 'await', 'null', 'undefined', 'true', 'false'],
            java: ['public', 'private', 'protected', 'class', 'interface', 'extends', 'implements', 'return', 'if', 'else', 'for', 'while', 'try', 'catch', 'finally', 'new', 'this', 'super', 'null', 'true', 'false'],
            bash: ['if', 'then', 'else', 'elif', 'fi', 'for', 'while', 'do', 'done', 'case', 'esac', 'function', 'return', 'exit', 'echo', 'export', 'source']
        };

        let highlighted = code;

        // Get keywords for language
        const langKeywords = keywords[lang] || [];

        if (langKeywords.length > 0) {
            // Highlight keywords (simple word boundary matching)
            langKeywords.forEach(keyword => {
                const regex = new RegExp(`\\b(${keyword})\\b`, 'g');
                highlighted = highlighted.replace(regex, `<span style="color: #0066cc; font-weight: 600;">$1</span>`);
            });
        }

        // Highlight strings
        highlighted = highlighted.replace(/(&quot;[^&]*&quot;|&#39;[^&#]*&#39;)/g, '<span style="color: #22863a;">$1</span>');

        // Highlight comments
        if (lang === 'python') {
            highlighted = highlighted.replace(/(#.*$)/gm, '<span style="color: #6a737d; font-style: italic;">$1</span>');
        } else if (lang === 'javascript' || lang === 'java') {
            highlighted = highlighted.replace(/(\/\/.*$)/gm, '<span style="color: #6a737d; font-style: italic;">$1</span>');
        } else if (lang === 'bash') {
            highlighted = highlighted.replace(/(#.*$)/gm, '<span style="color: #6a737d; font-style: italic;">$1</span>');
        }

        return highlighted;
    }

    // --------------------------------------------------------------------------
    // Utilities
    // --------------------------------------------------------------------------

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // --------------------------------------------------------------------------
    // Start Application
    // --------------------------------------------------------------------------

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
