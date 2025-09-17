const parser = require('@babel/parser');
const fs = require('fs');

function getPlugins(extension) {
    switch(extension) {
        case 'ts': return ['typescript'];
        case 'tsx': return ['typescript', 'jsx'];
        case 'jsx': return ['jsx'];
        default: return [];
    }
}

function parseCode(code, extension) {
    const ast = parser.parse(code, {
        sourceType: 'module',
        plugins: getPlugins(extension)
    });
    
    const result = { functions: [], classes: [], imports: [], exports: [] };
    
    function walk(node) {
        if (!node) return;
        
        switch (node.type) {
            case 'FunctionDeclaration':
            case 'ArrowFunctionExpression':
            case 'FunctionExpression':
                result.functions.push({
                    name: node.id?.name || 'anonymous',
                    range: [node.start, node.end],
                    line: node.loc.start.line
                });
                break;
                
            case 'ClassDeclaration':
                result.classes.push({
                    name: node.id.name,
                    range: [node.start, node.end],
                    line: node.loc.start.line
                });
                break;
                
            case 'ImportDeclaration':
                result.imports.push({
                    range: [node.start, node.end],
                    line: node.loc.start.line
                });
                break;
                
            case 'ExportDefaultDeclaration':
            case 'ExportNamedDeclaration':
                result.exports.push({
                    name: node.declaration?.id?.name || 'default',
                    range: [node.start, node.end],
                    line: node.loc.start.line
                });
                break;
        }
        
        for (const key in node) {
            const child = node[key];
            if (Array.isArray(child)) {
                child.forEach(walk);
            } else if (child?.type) {
                walk(child);
            }
        }
    }
    
    walk(ast);
    return result;
}

const inputFile = process.argv[2];
const request = JSON.parse(fs.readFileSync(inputFile, 'utf8'));
const result = parseCode(request.code, request.extension);
fs.writeFileSync(inputFile + '.result', JSON.stringify(result));