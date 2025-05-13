// apps/core/src/ai/llm.ts
import { logger } from '../utils/logger.js';
import { executeAgentLoop, streamAgentLoop } from './agentLoop.js';

export async function initializeLLM() {
  try {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error('Missing OPENAI_API_KEY in environment variables');
    }
    
    logger.info('Language model initialized successfully');
  } catch (error) {
    logger.error('Failed to initialize language model:', error);
    throw error;
  }
}

export async function getAssistantResponse(userInput: string) {
  try {
    logger.info('Executing agent loop for user input');
    const result = await executeAgentLoop(userInput, 10);
    logger.info(`Generated assistant response after ${result.steps} steps and ${result.toolResults.length} tool calls`);
    return result.finalText;
  } catch (error) {
    logger.error('Error generating assistant response:', error);
    return `I encountered an error while processing your request: ${error instanceof Error ? error.message : 'Unknown error'}. The full error has been logged for further investigation.`;
  }
}

export async function streamAssistantResponse(userInput: string) {
  try {
    logger.info('Initializing streaming agent loop');
    
    const textChunks: string[] = [];
    
    await streamAgentLoop(userInput, 10, {
      onText: (text) => {
        textChunks.push(text);
      },
      onToolCall: (toolName, args) => {
        logger.info(`Tool called: ${toolName} with args: ${JSON.stringify(args)}`);
      },
      onToolResult: (result) => {
        logger.info(`Tool result received: ${JSON.stringify(result).substring(0, 100)}...`);
      }
    });
    
    logger.info('Streaming agent loop completed');
    
    const encoder = new TextEncoder();
    return new ReadableStream({
      start(controller) {
        textChunks.forEach(chunk => {
          controller.enqueue(encoder.encode(chunk));
        });
        controller.close();
      }
    });
  } catch (error) {
    logger.error('Error streaming assistant response:', error);
    const encoder = new TextEncoder();
    const errorMessage = `I encountered an error while processing your request: ${error instanceof Error ? error.message : 'Unknown error'}. The full error has been logged for further investigation.`;
    
    return new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(errorMessage));
        controller.close();
      }
    });
  }
}