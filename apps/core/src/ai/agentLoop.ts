// apps/core/src/ai/agentLoop.ts
import { openai } from '@ai-sdk/openai';
import { generateText, streamText, type ToolCallUnion } from 'ai';
import { logger } from '../utils/logger.js';
import { SystemPrompt } from './prompt.js';
import { computerTools } from './tools.js';
import { captureScreen } from '../screen/capture.js';
import { resizeScreenshot } from '../screen/analyze.js';
import fs from 'fs';
import path from 'path';

type ToolCallResult = {
  toolName: string;
  args: Record<string, any>;
  result: any;
};

type AgentLoopCallbacks = {
  onThinking?: (thinking: string) => void;
  onToolCall?: (toolName: string, args: any) => void;
  onToolResult?: (result: any) => void;
  onText?: (text: string) => void;
  onComplete?: (finalText: string) => void;
};

export async function executeAgentLoop(
  userInput: string,
  maxSteps: number = 10,
  callbacks: AgentLoopCallbacks = {}
) {
  try {
    const screenshotInfo = await captureAutomaticScreenshot();
    let screenshotBuffer = fs.readFileSync(screenshotInfo.resizedScreenshotPath);
    
    let currentStep = 0;
    let toolResults: ToolCallResult[] = [];
    let finalText = '';
    
    while (currentStep < maxSteps) {
      logger.info(`Agent loop step ${currentStep + 1} of ${maxSteps}`);
      
      // Create messages with image content properly formatted
      const messages = [
        {
          role: 'user' as const,
          content: [
            { type: 'text', text: userInput },
            { type: 'image', image: screenshotBuffer }
          ]
        }
      ];
      
      const result = await generateText({
        model: openai('gpt-4o'),
        system: SystemPrompt,
        messages: messages,
        tools: computerTools,
        maxTokens: 1500,
        temperature: 0.7
      });
      
      finalText = result.text;
      if (callbacks.onText) {
        callbacks.onText(finalText);
      }
      
      if (result.toolCalls && result.toolCalls.length > 0) {
        for (const toolCall of result.toolCalls) {
          if (callbacks.onToolCall) {
            callbacks.onToolCall(toolCall.toolName, toolCall.args);
          }
          
          const toolName = toolCall.toolName;
          const args = toolCall.args;
          
          // Execute the tool (use the toolCall directly)
          const toolResult = await executeToolCall(toolName, args, computerTools);
          
          if (callbacks.onToolResult) {
            callbacks.onToolResult(toolResult);
          }
          
          toolResults.push({
            toolName: toolName,
            args: args,
            result: toolResult
          });
          
          // Take a new screenshot after each tool call to update the visual context
          const newScreenshotInfo = await captureAutomaticScreenshot();
          screenshotBuffer = fs.readFileSync(newScreenshotInfo.resizedScreenshotPath);
          
          // Update user input to include the tool result
          userInput = `I performed the action you requested using the ${toolName} tool. The result was: ${JSON.stringify(toolResult)}. What should I do next?`;
        }
        
        currentStep++;
        continue;
      }
      
      break;
    }
    
    if (callbacks.onComplete) {
      callbacks.onComplete(finalText);
    }
    
    return {
      finalText,
      toolResults,
      steps: currentStep
    };
  } catch (error) {
    logger.error('Error in agent loop:', error);
    throw error;
  }
}

export async function streamAgentLoop(
  userInput: string,
  maxSteps: number = 10,
  callbacks: AgentLoopCallbacks = {}
) {
  try {
    const screenshotInfo = await captureAutomaticScreenshot();
    let screenshotBuffer = fs.readFileSync(screenshotInfo.resizedScreenshotPath);
    
    let currentStep = 0;
    let toolResults: ToolCallResult[] = [];
    let finalText = '';
    let combinedText = '';
    
    while (currentStep < maxSteps) {
      logger.info(`Agent loop step ${currentStep + 1} of ${maxSteps}`);
      
      // Create messages with image content properly formatted
      const messages = [
        {
          role: 'user' as const,
          content: [
            { type: 'text', text: userInput },
            { type: 'image', image: screenshotBuffer }
          ]
        }
      ];
      
      const result = await streamText({
        model: openai('gpt-4o'),
        system: SystemPrompt,
        messages: messages,
        tools: computerTools,
        maxTokens: 1500,
        temperature: 0.7
      });
      
      // Collect streaming text
      combinedText = '';
      for await (const chunk of result.textStream) {
        combinedText += chunk;
        if (callbacks.onText) {
          callbacks.onText(chunk);
        }
      }
      
      finalText = combinedText;
      
      // Get the toolCalls that were generated
      const toolCallsArray = await result.toolCalls;
      
      if (toolCallsArray.length > 0) {
        for (const toolCall of toolCallsArray) {
          if (callbacks.onToolCall) {
            callbacks.onToolCall(toolCall.toolName, toolCall.args);
          }
          
          const toolName = toolCall.toolName;
          const args = toolCall.args;
          
          // Execute the tool
          const toolResult = await executeToolCall(toolName, args, computerTools);
          
          if (callbacks.onToolResult) {
            callbacks.onToolResult(toolResult);
          }
          
          toolResults.push({
            toolName: toolName,
            args: args,
            result: toolResult
          });
          
          // Take a new screenshot after each tool call to update the visual context
          const newScreenshotInfo = await captureAutomaticScreenshot();
          screenshotBuffer = fs.readFileSync(newScreenshotInfo.resizedScreenshotPath);
          
          // Update user input to include the tool result
          userInput = `I performed the action you requested using the ${toolName} tool. The result was: ${JSON.stringify(toolResult)}. What should I do next?`;
        }
        
        currentStep++;
        continue;
      }
      
      break;
    }
    
    if (callbacks.onComplete) {
      callbacks.onComplete(finalText);
    }
    
    return {
      finalText,
      toolResults,
      steps: currentStep
    };
  } catch (error) {
    logger.error('Error in streaming agent loop:', error);
    throw error;
  }
}

// Helper function to execute a tool call
async function executeToolCall(toolName: string, args: any, tools: any) {
  try {
    // Find the tool in the tools object
    const tool = tools[toolName];
    
    if (!tool) {
      logger.error(`Tool not found: ${toolName}`);
      return { error: `Tool not found: ${toolName}` };
    }
    
    // Execute the tool
    return await tool.execute(args);
  } catch (error) {
    logger.error(`Error executing tool ${toolName}:`, error);
    return { error: `Error executing tool: ${error instanceof Error ? error.message : 'Unknown error'}` };
  }
}

async function captureAutomaticScreenshot() {
  try {
    const screenshotsDir = path.join(process.cwd(), 'screenshots');
    fs.mkdirSync(screenshotsDir, { recursive: true });
    
    const timestamp = new Date().toISOString().replace(/:/g, '-');
    const filename = `screenshot-${timestamp}.png`;
    const filepath = path.join(screenshotsDir, filename);
    
    await captureScreen(undefined, filepath);
    
    const resizedPath = await resizeScreenshot(filepath, 1920, 1080);
    
    return {
      screenshotPath: filepath,
      resizedScreenshotPath: resizedPath,
      timestamp
    };
  } catch (error) {
    logger.error('Error capturing automatic screenshot:', error);
    throw error;
  }
}