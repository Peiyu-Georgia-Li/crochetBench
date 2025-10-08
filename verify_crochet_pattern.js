#!/usr/bin/env node

/**
 * Crochet Pattern Validator
 * 
 * This script validates crochet patterns using CrochetPARADE's actual parser (parse57.js).
 * It takes a pattern string as input and outputs whether it's valid or not.
 * USAGE:
 * node verify_crochet_pattern.js [pattern_string_or_file]
 *    or: cat pattern.txt | node verify_crochet_pattern.js
 * 
 * node verify_crochet_pattern.js example_pattern_work.txt
 * node verify_crochet_pattern.js example_pattern_not_work.txt
 */

// Import required modules
const fs = require('fs');
const path = require('path');

// Get the path to parse57.js
const parserPath = path.join(__dirname, 'CrochetPARADE', 'parse57.js');
// Check if parse57.js exists
if (!fs.existsSync(parserPath)) {
  console.error(`Error: Could not find the CrochetPARADE parser at ${parserPath}`);
  process.exit(1);
}

/**
 * Create a mock window object with required properties
 * to simulate the browser environment needed by parse57.js
 */
global.window = {
  alert: (msg) => { console.error(`Pattern Error: ${msg}`); }
};

// Mock console.trace to avoid stack trace output
const originalConsoleTrace = console.trace;
console.trace = () => {};

// Mock document object
global.document = {
  getElementById: () => ({
    value: '',
    style: {}
  }),
  createElement: () => ({
    style: {},
    appendChild: () => {}
  }),
  createTextNode: () => ({}),
  querySelector: () => ({ innerHTML: '' }),
  querySelectorAll: () => []
};

// Load the parser
const parserCode = fs.readFileSync(parserPath, 'utf8');

// Create a function context for the parser code
const funcContext = {
  window,
  document,
  console,
  setTimeout: setTimeout,
  clearTimeout: clearTimeout,
  localStorage: {
    getItem: () => null,
    setItem: () => {}
  },
  fetch: () => Promise.resolve({
    json: () => Promise.resolve({})
  })
};

// Execute the parser code in the global context to make functions available
eval(parserCode);

/**
 * Validate a crochet pattern
 * 
 * @param {string} patternString - The crochet pattern to validate
 * @return {object} - Validation result
 */
function validatePattern(patternString) {
  // Reset errors
  let errors = [];
  let warnings = [];
  
  // Override alert to capture errors
  window.alert = (msg) => {
    errors.push(msg);
  };
  
  // Override console.warn to capture warnings
  const originalConsoleWarn = console.warn;
  console.warn = (msg) => {
    warnings.push(msg);
  };
  
  try {
    // Check if the necessary functions are available from parse57.js
    if (typeof final !== 'function') {
      throw new Error('Could not access the required functions from parse57.js');
    }
    
    // Try to parse the pattern using the main parsing function from parse57.js
    const result = final(patternString);
    
    // Restore console functions
    console.warn = originalConsoleWarn;
    console.trace = originalConsoleTrace;
    
    // If parsing completed without errors, the pattern is valid
    if (errors.length === 0) {
      return {
        valid: true,
        message: 'Pattern is valid and can be compiled by CrochetPARADE',
        warnings: warnings.length > 0 ? warnings : undefined,
        details: result ? {
          stitchCount: result.Stitches ? result.Stitches.length : 0,
          rowCount: result.Stitches ? Math.max(...result.Stitches.map(s => s.nrow)) + 1 : 0
        } : undefined
      };
    } else {
      return {
        valid: false,
        errors: errors,
        warnings: warnings.length > 0 ? warnings : undefined
      };
    }
  } catch (error) {
    // Restore console functions
    console.warn = originalConsoleWarn;
    console.trace = originalConsoleTrace;
    
    return {
      valid: false,
      errors: errors.length > 0 ? errors : [`Exception during pattern parsing: ${error.message}`],
      exception: error.message
    };
  }
}

/**
 * Main function
 */
function main() {
  // Check if a pattern file was provided
  if (process.argv.length < 3) {
    // Check if we're receiving input from a pipe
    if (!process.stdin.isTTY) {
      let patternString = '';
      
      // Read from stdin
      process.stdin.on('data', (chunk) => {
        patternString += chunk;
      });
      
      process.stdin.on('end', () => {
        const result = validatePattern(patternString);
        console.log(JSON.stringify(result, null, 2));
        process.exit(result.valid ? 0 : 1);
      });
      
      return;
    }
    
    // No input provided
    console.log('Usage: node verify_crochet_pattern.js [pattern_string_or_file]');
    console.log('   or: cat pattern.txt | node verify_crochet_pattern.js');
    process.exit(1);
  }
  
  const arg = process.argv[2];
  
  // Check if the argument is a file path
  if (fs.existsSync(arg) && fs.statSync(arg).isFile()) {
    try {
      const patternString = fs.readFileSync(arg, 'utf8');
      const result = validatePattern(patternString);
      console.log(JSON.stringify(result, null, 2));
      process.exit(result.valid ? 0 : 1);
    } catch (error) {
      console.error(`Error reading file: ${error.message}`);
      process.exit(1);
    }
  } else {
    // Treat the argument as the pattern string
    const result = validatePattern(arg);
    console.log(JSON.stringify(result, null, 2));
    process.exit(result.valid ? 0 : 1);
  }
}

// Execute the main function
main();
