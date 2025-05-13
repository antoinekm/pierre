// apps/core/src/ai/conversation.ts
import { logger } from '../utils/logger.js';
import { executeAgentLoop } from './agentLoop.js';

type ConversationMessage = {
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
};

type ToolResult = {
  toolName: string;
  args: Record<string, any>;
  result: any;
  timestamp: number;
};

type Screenshot = {
  path: string;
  timestamp: number;
  description?: string;
};

type Conversation = {
  id: string;
  messages: ConversationMessage[];
  lastActivity: number;
  resources: {
    screenshots: Screenshot[];
    toolResults: ToolResult[];
  };
  context: Record<string, any>;
};

const conversations = new Map<string, Conversation>();
const CONVERSATION_TIMEOUT = 30 * 60 * 1000;

export function initializeConversationManager() {
  setInterval(() => {
    const now = Date.now();
    let cleanedCount = 0;
    
    for (const [id, conversation] of conversations.entries()) {
      if (now - conversation.lastActivity > CONVERSATION_TIMEOUT) {
        conversations.delete(id);
        cleanedCount++;
      }
    }
    
    if (cleanedCount > 0) {
      logger.info(`Cleaned up ${cleanedCount} inactive conversations`);
    }
  }, 5 * 60 * 1000);
  
  logger.info('Conversation manager initialized');
}

export function getOrCreateConversation(conversationId: string): Conversation {
  let conversation = conversations.get(conversationId);
  
  if (!conversation) {
    conversation = {
      id: conversationId,
      messages: [],
      lastActivity: Date.now(),
      resources: {
        screenshots: [],
        toolResults: []
      },
      context: {}
    };
    conversations.set(conversationId, conversation);
    logger.info(`Created new conversation: ${conversationId}`);
  }
  
  return conversation;
}

export function addConversationResource(
  conversationId: string, 
  resourceType: 'screenshot' | 'toolResult', 
  resource: any
): void {
  const conversation = getOrCreateConversation(conversationId);
  
  if (resourceType === 'screenshot') {
    conversation.resources.screenshots.push({
      path: resource.path,
      timestamp: Date.now(),
      description: resource.description || 'Screenshot'
    });
    
    logger.info(`Added screenshot to conversation ${conversationId}: ${resource.path}`);
  } else if (resourceType === 'toolResult') {
    conversation.resources.toolResults.push({
      ...resource,
      timestamp: Date.now()
    });
    
    logger.info(`Added tool result to conversation ${conversationId}: ${resource.toolName}`);
  }
  
  conversation.lastActivity = Date.now();
}

export function addConversationMessage(
  conversationId: string,
  role: 'user' | 'assistant',
  content: string
): void {
  const conversation = getOrCreateConversation(conversationId);
  
  conversation.messages.push({
    role,
    content,
    timestamp: Date.now()
  });
  
  conversation.lastActivity = Date.now();
  
  logger.info(`Added ${role} message to conversation ${conversationId}`);
}

export function updateConversationContext(
  conversationId: string,
  key: string,
  value: any
): void {
  const conversation = getOrCreateConversation(conversationId);
  
  conversation.context[key] = value;
  conversation.lastActivity = Date.now();
  
  logger.info(`Updated conversation context ${conversationId}: ${key}`);
}

export async function processConversationMessage(
  conversationId: string,
  userMessage: string
): Promise<string> {
  const conversation = getOrCreateConversation(conversationId);
  
  addConversationMessage(conversationId, 'user', userMessage);
  
  try {
    // Build context from previous messages
    let contextPrompt = userMessage;
    
    // Include relevant previous messages if available
    const recentMessages = conversation.messages.slice(-10);
    if (recentMessages.length > 1) {
      const previousExchange = recentMessages
        .slice(0, -1)
        .map(msg => `${msg.role === 'user' ? 'User' : 'You'}: ${msg.content}`)
        .join('\n\n');
        
      contextPrompt = `Previous conversation:\n${previousExchange}\n\nNew request: ${userMessage}`;
    }
    
    // If we have information about the last known state, include it
    if (conversation.context.lastKnownState) {
      contextPrompt = `${contextPrompt}\n\nPrevious screen state: ${conversation.context.lastKnownState}`;
    }
    
    const result = await executeAgentLoop(contextPrompt, 10, {
      onToolCall: (toolName, args) => {
        logger.info(`Tool called in conversation ${conversationId}: ${toolName}`);
      },
      onToolResult: (result) => {
        addConversationResource(conversationId, 'toolResult', result);
      }
    });
    
    // Update the last known state based on the final result
    if (result.finalText) {
      const stateDesc = result.finalText.length > 150 
        ? result.finalText.substring(0, 150) + '...'
        : result.finalText;
      
      updateConversationContext(conversationId, 'lastKnownState', `After your last actions, you ${stateDesc}`);
    }
    
    addConversationMessage(conversationId, 'assistant', result.finalText);
    
    return result.finalText;
  } catch (error) {
    logger.error(`Error processing message in conversation ${conversationId}:`, error);
    const errorMessage = `I encountered an error while processing your request: ${error instanceof Error ? error.message : 'Unknown error'}. The full error has been logged for further investigation.`;
    
    addConversationMessage(conversationId, 'assistant', errorMessage);
    
    return errorMessage;
  }
}

export function getConversation(conversationId: string): Conversation | undefined {
  return conversations.get(conversationId);
}