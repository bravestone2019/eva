import React, { useEffect, useRef, useState } from 'react';
import {
  AudioLines, Check, CircleHelp, Copy, LogIn, Pencil, Search,
  Menu, MessageSquarePlus, Mic, PanelLeftClose, PanelLeftOpen,
  Paperclip, Send, Settings2, Sparkles, ThumbsDown, ThumbsUp, Trash2,
  Square, X,
} from 'lucide-react';

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');
const AFFECT_SOCKET_BASE = API_BASE.replace(/^http/, 'ws');
const STORAGE_KEY = 'eva-chat-messages-v1';
const CONVERSATIONS_KEY = 'eva-conversations-v1';
const ACTIVE_CONVERSATION_KEY = 'eva-active-conversation-v1';
const AFFECT_FEEDBACK_KEY = 'eva-affect-feedback-v1';
const ADAPT_EMOTION_KEY = 'eva-adapt-emotion-v1';
const WAKE_ENABLED_KEY = 'eva-wake-enabled-v1';
const AUTH_TOKEN_KEY = 'eva-account-token-v1';
const SPEECH_MODE_KEY = 'eva-speech-mode-v1';
const TTS_MODE_KEY = 'eva-tts-mode-v1';
const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || '';
const SUPPORTED_AUDIO_EXTENSIONS = new Set(['wav', 'mp3', 'm4a', 'ogg', 'flac', 'webm']);
const SUPPORTED_TEXT_EXTENSIONS = new Set(['txt', 'md', 'csv', 'json', 'py', 'js', 'jsx', 'ts', 'tsx', 'html', 'css', 'yaml', 'yml', 'xml', 'log', 'sh', 'sql', 'java', 'c', 'cpp', 'h', 'go', 'rs', 'php', 'rb', 'toml', 'ini']);
const MAX_CHAT_ATTACHMENT_BYTES = 32 * 1024;

function isAllowedEmail(email) {
  return /^[^\s@]+@(gmail\.com|yahoo\.[a-z]{2,3}(?:\.[a-z]{2})?)$/i.test(email.trim());
}

function accountDisplayName(email) {
  const localPart = email?.split('@')[0]?.replace(/[._+-]+/g, ' ').trim();
  if (!localPart) return 'Account';
  return localPart.split(/\s+/).map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(' ');
}

function CodeChunk({ code, language }) {
  const [copied, setCopied] = useState(false);

  async function copyCode() {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  }

  return (
    <section className="assistant-code-block" aria-label={`${language || 'Code'} block`}>
      <div className="assistant-code-header">
        <span>{language || 'Code'}</span>
        <button type="button" onClick={copyCode} aria-label={copied ? 'Code copied' : 'Copy code'}>
          {copied ? <Check size={13} /> : <Copy size={13} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre><code>{code}</code></pre>
    </section>
  );
}

function renderInlineMarkdown(text, keyPrefix) {
  return text.split(/(\*\*[^*]+?\*\*)/g).map((part, index) => (
    part.startsWith('**') && part.endsWith('**')
      ? <strong key={`${keyPrefix}-bold-${index}`}>{part.slice(2, -2)}</strong>
      : part
  ));
}

function AssistantTextContent({ content }) {
  const nodes = [];
  let paragraphLines = [];
  let key = 0;
  const flushParagraph = () => {
    if (!paragraphLines.length) return;
    const lines = paragraphLines;
    const paragraphKey = `paragraph-${key++}`;
    nodes.push(
      <p className="assistant-md-paragraph" key={paragraphKey}>
        {lines.map((line, index) => (
          <React.Fragment key={`${paragraphKey}-line-${index}`}>
            {index > 0 && <br />}
            {renderInlineMarkdown(line, `${paragraphKey}-${index}`)}
          </React.Fragment>
        ))}
      </p>,
    );
    paragraphLines = [];
  };

  content.split('\n').forEach((line) => {
    const heading = line.match(/^\s*(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (heading) {
      flushParagraph();
      const level = heading[1].length;
      nodes.push(React.createElement(`h${level}`, {
        className: `assistant-md-heading assistant-md-heading-${level}`,
        key: `heading-${key++}`,
      }, renderInlineMarkdown(heading[2], `heading-${key}`)));
    } else if (!line.trim()) {
      flushParagraph();
    } else {
      paragraphLines.push(line);
    }
  });
  flushParagraph();
  return nodes;
}

function AssistantMessageContent({ content }) {
  const parts = [];
  const codeFence = /```([^\r\n]*)\r?\n([\s\S]*?)(?:```|$)/g;
  let cursor = 0;
  let match;
  while ((match = codeFence.exec(content)) !== null) {
    if (match.index > cursor) {
      parts.push({ type: 'text', value: content.slice(cursor, match.index) });
    }
    const language = match[1].trim().split(/\s+/)[0] || '';
    parts.push({ type: 'code', value: match[2].replace(/\r?\n$/, ''), language });
    cursor = codeFence.lastIndex;
  }
  if (cursor < content.length) parts.push({ type: 'text', value: content.slice(cursor) });
  if (!parts.length) parts.push({ type: 'text', value: content });

  return (
    <div className="assistant-content-flow">
      {parts.map((part, index) => part.type === 'code'
        ? <CodeChunk key={`code-${index}`} code={part.value} language={part.language} />
        : part.value && <div className="assistant-text-chunk" key={`text-${index}`}><AssistantTextContent content={part.value} /></div>)}
    </div>
  );
}

function readBooleanSetting(key, fallback) {
  try {
    const saved = localStorage.getItem(key);
    return saved === null ? fallback : saved === 'true';
  } catch {
    return fallback;
  }
}

function readAccountToken() {
  try { return localStorage.getItem(AUTH_TOKEN_KEY) || ''; } catch { return ''; }
}

function newConversation() {
  return {
    id: crypto.randomUUID(),
    title: 'New conversation',
    updatedAt: Date.now(),
    messages: [],
  };
}

function readSavedConversations() {
  try {
    const saved = JSON.parse(localStorage.getItem(CONVERSATIONS_KEY) || 'null');
    if (Array.isArray(saved) && saved.length) {
      const valid = saved.filter((conversation) => (
        conversation && typeof conversation.id === 'string'
        && typeof conversation.title === 'string'
        && Array.isArray(conversation.messages)
      ));
      if (valid.length) return valid;
    }
    const legacyMessages = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    if (Array.isArray(legacyMessages) && legacyMessages.length) {
      const firstQuestion = legacyMessages.find((message) => message.role === 'user')?.content;
      return [{
        ...newConversation(),
        title: firstQuestion?.slice(0, 40) || 'Previous conversation',
        messages: legacyMessages,
      }];
    }
  } catch {
    // Start with an empty conversation if browser storage is unavailable.
  }
  return [newConversation()];
}

function readActiveConversationId(conversations) {
  try {
    const savedId = localStorage.getItem(ACTIVE_CONVERSATION_KEY);
    if (conversations.some((conversation) => conversation.id === savedId)) return savedId;
  } catch {
    // Use the most recent local conversation when storage is unavailable.
  }
  return conversations[0].id;
}

function readAffectFeedback() {
  try {
    const saved = JSON.parse(localStorage.getItem(AFFECT_FEEDBACK_KEY) || '{}');
    return saved && typeof saved === 'object' && !Array.isArray(saved) ? saved : {};
  } catch {
    return {};
  }
}

const responseStyleNames = {
  concise_direct: 'concise and direct',
  step_by_step: 'step by step',
  warm_supportive: 'warm and supportive',
  encouraging: 'encouraging',
};

function EvaTreeMark({ size = 20 }) {
  return (
    <img
      className="eva-tree-mark"
      src="/eva-logo.png"
      width={size}
      height={size}
      alt=""
      aria-hidden="true"
    />
  );
}

export default function App() {
  const [initialConversations] = useState(readSavedConversations);
  const [conversations, setConversations] = useState(initialConversations);
  const [activeConversationId, setActiveConversationId] = useState(() => readActiveConversationId(initialConversations));
  const [showEmptyDraftInSpace, setShowEmptyDraftInSpace] = useState(() => {
    const activeId = readActiveConversationId(initialConversations);
    return initialConversations.find((conversation) => conversation.id === activeId)?.showInSpace === true;
  });
  const [messages, setMessages] = useState(() => (
    initialConversations.find((conversation) => conversation.id === readActiveConversationId(initialConversations))
      || initialConversations[0]
  ).messages);
  const [draft, setDraft] = useState('');
  const [pendingAttachment, setPendingAttachment] = useState(null);
  const [busy, setBusy] = useState(false);
  const [recording, setRecording] = useState(false);
  const [listening, setListening] = useState(false);
  const [voiceMode, setVoiceMode] = useState(false);
  const [affectStreamActive, setAffectStreamActive] = useState(false);
  const [affectFeedback, setAffectFeedback] = useState(readAffectFeedback);
  const [adaptEmotion, setAdaptEmotion] = useState(() => readBooleanSetting(ADAPT_EMOTION_KEY, true));
  const [wakeEnabled, setWakeEnabled] = useState(() => readBooleanSetting(WAKE_ENABLED_KEY, true));
  const [speechMode, setSpeechMode] = useState(() => { try { return localStorage.getItem(SPEECH_MODE_KEY) || 'browser'; } catch { return 'browser'; } });
  const [ttsMode, setTtsMode] = useState(() => { try { return localStorage.getItem(TTS_MODE_KEY) || 'edge'; } catch { return 'edge'; } });
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(false);
  const [accountToken, setAccountToken] = useState(readAccountToken);
  const [account, setAccount] = useState(null);
  const [accountReady, setAccountReady] = useState(false);
  const [accountDialogOpen, setAccountDialogOpen] = useState(false);
  const [signOutConfirmOpen, setSignOutConfirmOpen] = useState(false);
  const [signOutBusy, setSignOutBusy] = useState(false);
  const [signOutError, setSignOutError] = useState('');
  const [accountMode, setAccountMode] = useState('login');
  const [accountEmail, setAccountEmail] = useState('');
  const [accountPassword, setAccountPassword] = useState('');
  const [authError, setAuthError] = useState('');
  const [authNotice, setAuthNotice] = useState('');
  const [authBusy, setAuthBusy] = useState(false);
  const [knowledgeDocuments, setKnowledgeDocuments] = useState([]);
  const [documentBusy, setDocumentBusy] = useState(false);
  const [documentError, setDocumentError] = useState('');
  const [error, setError] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(258);
  const [sidebarResizing, setSidebarResizing] = useState(false);
  const [spaceSearch, setSpaceSearch] = useState('');
  const [spaceSearchOpen, setSpaceSearchOpen] = useState(false);
  const [renamingConversationId, setRenamingConversationId] = useState(null);
  const [renameDraft, setRenameDraft] = useState('');
  const conversationsRef = useRef(conversations);
  conversationsRef.current = conversations;
  const renameInputRef = useRef(null);
  const renameFormRef = useRef(null);
  const spaceSearchRef = useRef(null);
  const sidebarDragRef = useRef(null);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const knowledgeInputRef = useRef(null);
  const googleButtonRef = useRef(null);
  const aboutButtonRef = useRef(null);
  const aboutCloseRef = useRef(null);
  const signOutConfirmRef = useRef(null);
  const recorderRef = useRef(null);
  const recognitionRef = useRef(null);
  const streamRef = useRef(null);
  const silenceTimerRef = useRef(null);
  const silenceFrameRef = useRef(null);
  const audioContextRef = useRef(null);
  const chunksRef = useRef([]);
  const messagesRef = useRef(messages);
  const busyRef = useRef(false);
  const speechBaseRef = useRef('');
  const speechFinalRef = useRef('');
  const voiceModeRef = useRef(false);
  const restartListeningTimerRef = useRef(null);
  const audioCancelRef = useRef(null);
  const ttsAbortControllerRef = useRef(null);
  const speechPlaybackRef = useRef(null);
  const assistantSpeechTextRef = useRef('');
  const bargeInRef = useRef(false);
  const turnRecorderRef = useRef(null);
  const turnChunksRef = useRef([]);
  const turnStreamRef = useRef(null);
  const turnCapturePromiseRef = useRef(null);
  const affectSocketRef = useRef(null);
  const affectAudioContextRef = useRef(null);
  const affectSourceRef = useRef(null);
  const affectWorkletRef = useRef(null);
  const affectMuteRef = useRef(null);
  const affectQueuedFramesRef = useRef([]);
  const affectFinalResolveRef = useRef(null);
  const affectFlushResolveRef = useRef(null);
  const liveAffectScoresRef = useRef(null);
  const wakeRecognitionRef = useRef(null);
  const wakeEnabledRef = useRef(true);
  const wakePendingRef = useRef(false);
  const wakeChatIdRef = useRef(null);
  const wakePhraseBufferRef = useRef('');
  const wakePhraseResetTimerRef = useRef(null);
  const wakeActivationTimerRef = useRef(null);
  const wakeRestartTimerRef = useRef(null);
  const wakeActionsRef = useRef(null);
  const restartVoiceAfterNewChatRef = useRef(false);
  const silenceTimeoutMs = 2000;
  const initialVoiceTimeoutMs = 5000;

  function clearSilenceTimer() {
    if (silenceTimerRef.current !== null) {
      window.clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
  }

  function stopSilenceMonitor() {
    clearSilenceTimer();
    if (silenceFrameRef.current !== null) {
      window.cancelAnimationFrame(silenceFrameRef.current);
      silenceFrameRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
  }

  function scheduleVoiceListening() {
    if (!voiceModeRef.current || recognitionRef.current || recording) return;
    if (restartListeningTimerRef.current !== null) {
      window.clearTimeout(restartListeningTimerRef.current);
    }
    restartListeningTimerRef.current = window.setTimeout(() => {
      restartListeningTimerRef.current = null;
      if (voiceModeRef.current && !recognitionRef.current && !recording) startLiveDictation({ autoSend: true });
    }, 350);
  }

  function scheduleWakePhraseListening(delay = 450) {
    if (!wakeEnabledRef.current) return;
    if (wakeRestartTimerRef.current !== null) window.clearTimeout(wakeRestartTimerRef.current);
    wakeRestartTimerRef.current = window.setTimeout(() => {
      wakeRestartTimerRef.current = null;
      startWakePhraseListener();
    }, delay);
  }

  function stopWakePhraseListener() {
    if (wakeRestartTimerRef.current !== null) {
      window.clearTimeout(wakeRestartTimerRef.current);
      wakeRestartTimerRef.current = null;
    }
    if (wakePhraseResetTimerRef.current !== null) {
      window.clearTimeout(wakePhraseResetTimerRef.current);
      wakePhraseResetTimerRef.current = null;
    }
    if (wakeActivationTimerRef.current !== null) {
      window.clearTimeout(wakeActivationTimerRef.current);
      wakeActivationTimerRef.current = null;
    }
    wakePhraseBufferRef.current = '';
    if (wakeRecognitionRef.current) {
      const recognition = wakeRecognitionRef.current;
      wakeRecognitionRef.current = null;
      recognition.onresult = null;
      recognition.onerror = null;
      recognition.onend = null;
      recognition.abort();
    }
  }

  function startWakePhraseListener() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!wakeEnabledRef.current || voiceModeRef.current || wakeRecognitionRef.current) return;
    if (!SpeechRecognition) {
      wakeEnabledRef.current = false;
      setWakeEnabled(false);
      setError('Automatic wake phrases are not supported in this browser. Try Chrome or Edge.');
      return;
    }

    try {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = false;
      recognition.lang = navigator.language || 'en-US';
      wakePhraseBufferRef.current = '';
      recognition.onresult = (event) => {
        const finalParts = [];
        for (let index = event.resultIndex; index < event.results.length; index += 1) {
          if (!event.results[index].isFinal) continue;
          finalParts.push(event.results[index][0].transcript);
        }
        if (!finalParts.length) return;
        const heard = finalParts.join(' ').toLowerCase().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim();
        wakePhraseBufferRef.current = `${wakePhraseBufferRef.current} ${heard}`.trim().split(' ').slice(-40).join(' ');
        if (wakePhraseResetTimerRef.current !== null) window.clearTimeout(wakePhraseResetTimerRef.current);
        wakePhraseResetTimerRef.current = window.setTimeout(() => {
          wakePhraseBufferRef.current = '';
          wakePhraseResetTimerRef.current = null;
        }, 3500);

        const phrase = wakePhraseBufferRef.current;
        const openedChat = findConversationFromVoicePhrase(phrase);
        const startsNewChat = isNewConversationPhrase(phrase);
        // Match the full phrase across recognition result chunks so a pause
        // between "open the chat" and its title does not lose the command.
        if (openedChat || startsNewChat) {
          if (wakeActivationTimerRef.current !== null) {
            window.clearTimeout(wakeActivationTimerRef.current);
            wakeActivationTimerRef.current = null;
          }
          wakeChatIdRef.current = openedChat?.id || null;
          wakePendingRef.current = true;
          recognition.stop();
        } else if (/\beva\b/.test(phrase) && wakeActivationTimerRef.current === null) {
          // Give a command such as "EVA, open the chat ..." a moment to
          // arrive in the next recognition result before starting a new chat.
          wakeActivationTimerRef.current = window.setTimeout(() => {
            wakeActivationTimerRef.current = null;
            wakeChatIdRef.current = null;
            wakePendingRef.current = true;
            recognition.stop();
          }, 1000);
        }
      };
      recognition.onerror = (event) => {
        if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
          wakeEnabledRef.current = false;
          setWakeEnabled(false);
          setError('Microphone or speech access was blocked. Allow access in your browser site settings, then refresh EVA.');
        } else if (event.error !== 'aborted' && event.error !== 'no-speech') {
          setError(`Wake phrase listening stopped (${event.error}). Check microphone permission and refresh EVA.`);
        }
      };
      recognition.onend = () => {
        if (wakeRecognitionRef.current === recognition) wakeRecognitionRef.current = null;
        if (wakePendingRef.current) {
          wakePendingRef.current = false;
          if (wakeEnabledRef.current && !voiceModeRef.current) {
            const chatId = wakeChatIdRef.current;
            wakeChatIdRef.current = null;
            if (chatId) wakeActionsRef.current?.openConversation(chatId);
            else {
              wakeActionsRef.current?.startNewChat();
              wakeActionsRef.current?.startVoiceConversation();
            }
          }
          return;
        }
        if (wakeEnabledRef.current && !voiceModeRef.current) {
          scheduleWakePhraseListening(350);
        }
      };
      recognition.start();
      wakeRecognitionRef.current = recognition;
      setError('');
    } catch {
      wakeEnabledRef.current = false;
      setWakeEnabled(false);
      setError('Could not start automatic wake phrase listening. Check microphone permission and refresh EVA.');
    }
  }

  function isEndConversationPhrase(text) {
    const normalized = text.toLowerCase().replace(/[^a-z0-9\s']/g, ' ').replace(/\s+/g, ' ').trim();
    return /^(?:eva\s+)?(?:(?:please\s+)?(?:end|stop|finish|close)(?:\s+(?:of|the|our|this))?\s+(?:conversation|chat)|end\s+of\s+(?:the\s+)?conversation|goodbye(?:\s+eva)?|bye\s+eva|that's all(?: for now)?)(?:\s+now)?$/.test(normalized);
  }

  function isNewConversationPhrase(text) {
    const normalized = text.toLowerCase().replace(/[^a-z0-9\s']/g, ' ').replace(/\s+/g, ' ').trim();
    return /\b(?:open|start|begin|create|make|switch to|take me to|go to)\b.*\b(?:new|another)\b.*\b(?:conversation|chat)\b/.test(normalized)
      || /\b(?:new|another)\s+(?:conversation|chat)\b/.test(normalized);
  }

  function findConversationFromVoicePhrase(text) {
    const normalize = (value) => value.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim();
    const normalized = normalize(text);
    const isOpenCommand = /\b(?:open|find|search|show|switch to|go to|take me to)\b/.test(normalized);
    if (!isOpenCommand || isNewConversationPhrase(normalized)) return null;

    const commandWords = new Set([
      'open', 'find', 'search', 'show', 'switch', 'to', 'go', 'take', 'me', 'the', 'chat',
      'conversation', 'called', 'named', 'titled', 'please', 'eva', 'hey', 'hello', 'hi',
      'my', 'for', 'with', 'about', 'can', 'you', 'could', 'would', 'up',
    ]);
    const queryText = normalized
      .replace(/\b(?:search for|show me|take me to|switch to|go to)\b/g, ' ')
      .replace(/\b(?:open|find|search|show|switch|go|take|me|the|chat|conversation|called|named|titled|please|eva|hey|hello|hi|my|for|with|about|can|you|could|would)\b/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
    const queryWords = [...new Set(queryText.split(' ').filter((word) => word.length > 1 && !commandWords.has(word)))];
    if (!queryText || !queryWords.length) return null;

    const candidates = conversationsRef.current
      .filter((conversation) => conversation.messages.length > 0)
      .map((conversation) => ({
        conversation,
        title: normalize(conversation.title),
        searchableText: normalize([
          conversation.title,
          conversation.messages.find((message) => message.role === 'user')?.content || '',
        ].join(' ')),
      }))
      .map((candidate) => {
        const terms = new Set(candidate.searchableText.split(' ').filter((word) => word.length > 1));
        const matched = queryWords.filter((word) => terms.has(word));
        const exactPhrase = queryText.length >= 3 && candidate.searchableText.includes(queryText);
        return {
          ...candidate,
          score: exactPhrase ? 2 : matched.length / queryWords.length,
          matchedCount: matched.length,
        };
      })
      .filter((candidate) => candidate.score === 2 || candidate.matchedCount >= 2);
    candidates.sort((first, second) => second.score - first.score || second.matchedCount - first.matchedCount);
    const best = candidates[0];
    const runnerUp = candidates[1];
    if (!best || (runnerUp && best.score !== 2 && best.score - runnerUp.score < 0.2)) return null;
    return best.conversation;
  }

  function isAssistantSpeechEcho(text) {
    const normalize = (value) => value.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim();
    const spoken = normalize(text);
    if (spoken.split(' ').length < 2) return false;
    return normalize(assistantSpeechTextRef.current).includes(spoken);
  }

  function interruptAssistantSpeech() {
    if (!bargeInRef.current) return;
    bargeInRef.current = false;
    assistantSpeechTextRef.current = '';
    ttsAbortControllerRef.current?.abort();
    audioCancelRef.current?.();
  }

  async function speakAssistantResponse(text) {
    if (speechPlaybackRef.current) return speechPlaybackRef.current;
    if (voiceModeRef.current) {
      assistantSpeechTextRef.current = text;
      bargeInRef.current = true;
      // Keep recognition open during playback. Echo-like transcripts matching
      // EVA's own reply are ignored; a different utterance interrupts it.
      startLiveDictation({ autoSend: true, bargeIn: true });
    }
    if (ttsMode === 'device') {
      const playback = new Promise((resolve, reject) => {
        if (!window.speechSynthesis || !window.SpeechSynthesisUtterance) { reject(new Error('Device speech is unavailable.')); return; }
        const utterance = new SpeechSynthesisUtterance(text);
        const voices = window.speechSynthesis.getVoices();
        const preferred = voices.find((voice) => voice.lang?.toLowerCase().startsWith('en') && /aria|zira|jenny|samantha|female|hazel|susan/i.test(voice.name))
          || voices.find((voice) => voice.lang?.toLowerCase().startsWith('en'));
        if (preferred) utterance.voice = preferred;
        utterance.pitch = 1.06;
        utterance.rate = 1;
      utterance.onend = () => { audioCancelRef.current = null; resolve(); };
      utterance.onerror = () => { audioCancelRef.current = null; reject(new Error('Device speech playback failed.')); };
      audioCancelRef.current = () => { window.speechSynthesis.cancel(); audioCancelRef.current = null; resolve(); };
        window.speechSynthesis.speak(utterance);
      });
      speechPlaybackRef.current = playback;
      try { await playback; } catch { if (voiceModeRef.current) setError('Device speech is unavailable. Choose Aria Neural or use a browser with installed voices.'); }
      finally {
        if (speechPlaybackRef.current === playback) speechPlaybackRef.current = null;
        bargeInRef.current = false;
        assistantSpeechTextRef.current = '';
      }
      return;
    }
    const controller = new AbortController();
    ttsAbortControllerRef.current = controller;
    const playback = (async () => {
      const response = await fetch(`${API_BASE}/v1/tts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error('Speech service unavailable');
      const url = URL.createObjectURL(await response.blob());
      if (!voiceModeRef.current) {
        URL.revokeObjectURL(url);
        return;
      }
      await new Promise((resolve, reject) => {
        const player = new Audio(url);
        let finished = false;
        const finish = (error) => {
          if (finished) return;
          finished = true;
          if (audioCancelRef.current === cancel) audioCancelRef.current = null;
          URL.revokeObjectURL(url);
          if (error) reject(error);
          else resolve();
        };
        const cancel = () => {
          player.pause();
          finish();
        };
        audioCancelRef.current = cancel;
        player.onended = finish;
        player.onerror = () => finish(new Error('Unable to play generated speech'));
        player.play().catch((error) => finish(error));
      });
    })();
    speechPlaybackRef.current = playback;
    try {
      await playback;
    } catch (error) {
      if (error.name !== 'AbortError' && voiceModeRef.current) {
        setError('EVA could not play the voice reply. Check the connection and try again.');
      }
    } finally {
      if (speechPlaybackRef.current === playback) speechPlaybackRef.current = null;
      if (ttsAbortControllerRef.current === controller) ttsAbortControllerRef.current = null;
      bargeInRef.current = false;
      assistantSpeechTextRef.current = '';
    }
  }

  function watchForAudioSilence(stream, onSilence) {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) {
      silenceTimerRef.current = window.setTimeout(onSilence, silenceTimeoutMs);
      return;
    }

    const context = new AudioContext();
    audioContextRef.current = context;
    context.resume().catch(() => {});
    const analyser = context.createAnalyser();
    analyser.fftSize = 512;
    context.createMediaStreamSource(stream).connect(analyser);
    const samples = new Uint8Array(analyser.fftSize);
    let lastVoiceAt = performance.now();
    const listeningStartedAt = lastVoiceAt;
    let heardVoice = false;
    const checkLevel = () => {
      analyser.getByteTimeDomainData(samples);
      let energy = 0;
      for (const sample of samples) {
        const amplitude = (sample - 128) / 128;
        energy += amplitude * amplitude;
      }
      const rms = Math.sqrt(energy / samples.length);
      const now = performance.now();
      if (rms > 0.025) {
        heardVoice = true;
        lastVoiceAt = now;
      }
      const timeout = heardVoice ? silenceTimeoutMs : initialVoiceTimeoutMs;
      const referenceTime = heardVoice ? lastVoiceAt : listeningStartedAt;
      if (now - referenceTime >= timeout) {
        stopSilenceMonitor();
        onSilence();
        return;
      }
      silenceFrameRef.current = window.requestAnimationFrame(checkLevel);
    };
    silenceFrameRef.current = window.requestAnimationFrame(checkLevel);
  }

  useEffect(() => {
    messagesRef.current = messages;
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages]);

  useEffect(() => {
    if (renamingConversationId) {
      renameInputRef.current?.focus();
      renameInputRef.current?.select();
    }
  }, [renamingConversationId]);

  useEffect(() => {
    if (!renamingConversationId) return undefined;
    const finishRenameOutside = (event) => {
      if (renameFormRef.current?.contains(event.target)) return;
      saveConversationRename(renamingConversationId);
    };
    document.addEventListener('pointerdown', finishRenameOutside);
    return () => document.removeEventListener('pointerdown', finishRenameOutside);
  }, [renamingConversationId, renameDraft]);

  useEffect(() => {
    if (!spaceSearchOpen) return undefined;
    const closeSearchOutside = (event) => {
      if (spaceSearchRef.current?.contains(event.target)) return;
      setSpaceSearchOpen(false);
      setSpaceSearch('');
    };
    document.addEventListener('pointerdown', closeSearchOutside);
    return () => document.removeEventListener('pointerdown', closeSearchOutside);
  }, [spaceSearchOpen]);

  useEffect(() => {
    setConversations((current) => current.map((conversation) => {
      if (conversation.id !== activeConversationId) return conversation;
      const firstQuestion = messages.find((message) => message.role === 'user')?.content;
      return {
        ...conversation,
        title: conversation.title === 'New conversation' && firstQuestion
          ? firstQuestion.slice(0, 40)
          : conversation.title,
        updatedAt: Date.now(),
        messages,
      };
    }));
  }, [activeConversationId, messages]);

  useEffect(() => {
    try {
      localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(conversations));
      localStorage.setItem(ACTIVE_CONVERSATION_KEY, activeConversationId);
    } catch {
      setError('Browser storage is unavailable. This conversation may not be saved.');
    }
  }, [conversations, activeConversationId]);

  useEffect(() => {
    if (!accountToken) {
      setAccountReady(true);
      return undefined;
    }
    let cancelled = false;
    setAccountReady(false);
    const headers = { Authorization: `Bearer ${accountToken}` };
    Promise.all([
      fetch(`${API_BASE}/v1/auth/me`, { headers }),
      fetch(`${API_BASE}/v1/conversations`, { headers }),
    ]).then(async ([profileResponse, chatsResponse]) => {
      if (!profileResponse.ok || !chatsResponse.ok) throw new Error('Your session expired. Sign in again.');
      const profile = await profileResponse.json();
      const serverChats = (await chatsResponse.json()).conversations || [];
      if (cancelled) return;
      setAccount(profile.user);
      const localChats = conversations.filter((chat) => chat.messages?.length);
      const serverIds = new Set(serverChats.map((chat) => chat.id));
      const mergedChats = [...serverChats, ...localChats.filter((chat) => !serverIds.has(chat.id))]
        .sort((first, second) => second.updatedAt - first.updatedAt);
      const next = mergedChats.find((chat) => chat.id === activeConversationId) || mergedChats[0];
      setConversations(mergedChats.length ? mergedChats : [newConversation()]);
      if (next) { setActiveConversationId(next.id); setMessages(next.messages || []); }
      else { const fresh = newConversation(); setActiveConversationId(fresh.id); setMessages([]); }
      setAccountReady(true);
    }).catch((loadError) => {
      if (cancelled) return;
      localStorage.removeItem(AUTH_TOKEN_KEY);
      setAccountToken('');
      setAccount(null);
      setAccountReady(true);
      setError(loadError.message);
    });
    return () => { cancelled = true; };
  }, [accountToken]);

  useEffect(() => {
    if (!accountToken || !accountReady || !account) return;
    const timer = window.setTimeout(() => {
      const headers = { Authorization: `Bearer ${accountToken}`, 'Content-Type': 'application/json' };
      conversations.forEach((conversation) => {
        if (!conversation.messages?.length) return;
        fetch(`${API_BASE}/v1/conversations/${encodeURIComponent(conversation.id)}`, {
          method: 'PUT', headers, body: JSON.stringify(conversation),
        }).then((response) => {
          if (!response.ok) setError('Could not sync chats to your account.');
        }).catch(() => setError('Could not sync chats to your account.'));
      });
    }, 500);
    return () => window.clearTimeout(timer);
  }, [conversations, accountToken, accountReady, account]);

  useEffect(() => {
    try {
      localStorage.setItem(AFFECT_FEEDBACK_KEY, JSON.stringify(affectFeedback));
    } catch {
      // Reply ratings stay optional when browser storage is unavailable.
    }
  }, [affectFeedback]);

  useEffect(() => {
    try {
      localStorage.setItem(ADAPT_EMOTION_KEY, String(adaptEmotion));
    } catch {
      // Preference still applies for this session if storage is unavailable.
    }
  }, [adaptEmotion]);

  useEffect(() => {
    wakeEnabledRef.current = wakeEnabled;
    try {
      localStorage.setItem(WAKE_ENABLED_KEY, String(wakeEnabled));
    } catch {
      // Preference still applies for this session if storage is unavailable.
    }
    if (wakeEnabled) startWakePhraseListener();
    else stopWakePhraseListener();
  }, [wakeEnabled]);

  useEffect(() => { if (settingsOpen) refreshKnowledgeDocuments(); }, [settingsOpen]);
  useEffect(() => {
    if (!aboutOpen) return undefined;
    aboutCloseRef.current?.focus();
    const closeOnEscape = (event) => {
      if (event.key === 'Escape') {
        setAboutOpen(false);
        aboutButtonRef.current?.focus();
      }
    };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [aboutOpen]);
  useEffect(() => {
    if (!signOutConfirmOpen) return undefined;
    signOutConfirmRef.current?.focus();
    const closeOnEscape = (event) => {
      if (event.key === 'Escape' && !signOutBusy) setSignOutConfirmOpen(false);
    };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [signOutConfirmOpen, signOutBusy]);
  useEffect(() => {
    try { localStorage.setItem(SPEECH_MODE_KEY, speechMode); localStorage.setItem(TTS_MODE_KEY, ttsMode); } catch { /* session preferences remain active */ }
  }, [speechMode, ttsMode]);

  useEffect(() => () => {
    voiceModeRef.current = false;
    if (restartListeningTimerRef.current !== null) window.clearTimeout(restartListeningTimerRef.current);
    if (wakeRestartTimerRef.current !== null) window.clearTimeout(wakeRestartTimerRef.current);
    if (wakePhraseResetTimerRef.current !== null) window.clearTimeout(wakePhraseResetTimerRef.current);
    if (wakeActivationTimerRef.current !== null) window.clearTimeout(wakeActivationTimerRef.current);
    if (wakeRecognitionRef.current) {
      wakeRecognitionRef.current.onresult = null;
      wakeRecognitionRef.current.onerror = null;
      wakeRecognitionRef.current.onend = null;
      wakeRecognitionRef.current.abort();
      wakeRecognitionRef.current = null;
    }
    if (recognitionRef.current) {
      recognitionRef.current.onresult = null;
      recognitionRef.current.onerror = null;
      recognitionRef.current.onend = null;
      recognitionRef.current.abort();
    }
    if (recorderRef.current?.state === 'recording') {
      recorderRef.current.onstop = null;
      recorderRef.current.stop();
    }
    stopSilenceMonitor();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    if (turnRecorderRef.current?.state === 'recording') turnRecorderRef.current.stop();
    if (affectWorkletRef.current) {
      affectWorkletRef.current.port.onmessage = null;
      affectWorkletRef.current.disconnect();
    }
    affectSourceRef.current?.disconnect();
    affectMuteRef.current?.disconnect();
    affectAudioContextRef.current?.close().catch(() => {});
    if (affectSocketRef.current && affectSocketRef.current.readyState < WebSocket.CLOSING) {
      affectSocketRef.current.close();
    }
    turnStreamRef.current?.getTracks().forEach((track) => track.stop());
  }, []);

  function startNewChat() {
    stopVoiceConversation({ discard: true, resumeWake: false });
    if (recognitionRef.current) {
      recognitionRef.current.onresult = null;
      recognitionRef.current.onerror = null;
      recognitionRef.current.onend = null;
      recognitionRef.current.abort();
    }
    if (recorderRef.current?.state === 'recording') {
      recorderRef.current.onstop = null;
      recorderRef.current.stop();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      setRecording(false);
    }
    stopSilenceMonitor();
    recognitionRef.current = null;
    speechBaseRef.current = '';
    speechFinalRef.current = '';
    setListening(false);
    const conversation = { ...newConversation(), showInSpace: true };
    setConversations((current) => [
      conversation,
      ...current.filter((item) => item.id !== activeConversationId || messages.length > 0),
    ]);
    setActiveConversationId(conversation.id);
    setMessages([]);
    setShowEmptyDraftInSpace(true);
    setDraft('');
    setError('');
    setSidebarOpen(false);
  }

  useEffect(() => {
    const handleShortcut = (event) => {
      if (event.altKey && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        startNewChat();
      }
    };
    window.addEventListener('keydown', handleShortcut);
    return () => window.removeEventListener('keydown', handleShortcut);
  }, [activeConversationId, conversations]);

  function openConversation(conversationId) {
    if (conversationId === activeConversationId) {
      setSidebarOpen(false);
      return;
    }
    const conversation = conversations.find((item) => item.id === conversationId);
    if (!conversation) return;
    stopVoiceConversation({ discard: true });
    if (recognitionRef.current) {
      recognitionRef.current.onresult = null;
      recognitionRef.current.onerror = null;
      recognitionRef.current.onend = null;
      recognitionRef.current.abort();
      recognitionRef.current = null;
    }
    if (recorderRef.current?.state === 'recording') {
      recorderRef.current.onstop = null;
      recorderRef.current.stop();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      setRecording(false);
    }
    stopSilenceMonitor();
    setListening(false);
    setActiveConversationId(conversation.id);
    setMessages(conversation.messages);
    setShowEmptyDraftInSpace(false);
    setDraft('');
    setError('');
    setSidebarOpen(false);
  }

  function beginRenameConversation(conversation) {
    setRenamingConversationId(conversation.id);
    setRenameDraft(conversation.title);
  }

  function cancelRenameConversation() {
    setRenamingConversationId(null);
    setRenameDraft('');
  }

  function saveConversationRename(conversationId) {
    const title = renameDraft.trim();
    if (!title) {
      cancelRenameConversation();
      return;
    }
    setConversations((current) => current.map((conversation) => (
      conversation.id === conversationId
        ? { ...conversation, title, updatedAt: Date.now() }
        : conversation
    )));
    cancelRenameConversation();
  }

  function deleteConversation(conversationId) {
    if (accountToken) {
      fetch(`${API_BASE}/v1/conversations/${encodeURIComponent(conversationId)}`, {
        method: 'DELETE', headers: { Authorization: `Bearer ${accountToken}` },
      }).catch(() => setError('Could not delete the synced conversation.'));
    }
    const remaining = conversations.filter((conversation) => conversation.id !== conversationId);
    if (conversationId !== activeConversationId) {
      setConversations(remaining);
      return;
    }

    stopVoiceConversation({ discard: true });
    const nextConversation = remaining[0] || newConversation();
    setConversations(remaining.length ? remaining : [nextConversation]);
    setActiveConversationId(nextConversation.id);
    setMessages(nextConversation.messages);
    setShowEmptyDraftInSpace(false);
    setDraft('');
    setError('');
  }

  function rateReply(message, rating) {
    setAffectFeedback((current) => ({
      ...current,
      [message.id]: {
        rating,
        adapted: Boolean(message.affect?.adapted),
        response_style: message.affect?.response_style || 'standard',
        confidence: Number(message.affect?.confidence || 0),
        latency_ms: Number(message.latencyMs || 0),
        rated_at: new Date().toISOString(),
      },
    }));
  }

  function rateReplyReason(messageId, reason) {
    setAffectFeedback((current) => ({
      ...current,
      [messageId]: { ...current[messageId], reason },
    }));
  }

  async function submitAccount(event) {
    event.preventDefault();
    setAuthError('');
    setAuthNotice('');
    if (!isAllowedEmail(accountEmail)) {
      setAuthError('Use a Gmail or Yahoo email address.');
      return;
    }
    if (accountMode === 'register' && !/^\d{6}$/.test(accountPassword)) {
      setAuthError('Choose a password with exactly 6 digits.');
      return;
    }
    setAuthBusy(true);
    try {
      const response = await fetch(`${API_BASE}/v1/auth/${accountMode === 'register' ? 'register' : 'login'}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: accountEmail, password: accountPassword }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || 'Could not sign in.');
      if (accountMode === 'register') {
        setAccountEmail(result.user.email || accountEmail);
        setAccountPassword('');
        setAccountMode('login');
        setAuthNotice('Your account is ready. Sign in with your email and password to continue.');
        return;
      }
      localStorage.setItem(AUTH_TOKEN_KEY, result.token);
      setAccount(result.user);
      setAccountDialogOpen(false);
      setAccountPassword('');
      setAccountReady(false);
      setAccountToken(result.token);
    } catch (authError) {
      setAuthError(authError.message || 'Account service is unavailable.');
    } finally {
      setAuthBusy(false);
    }
  }

  useEffect(() => {
    if (!accountDialogOpen || account || !GOOGLE_CLIENT_ID) return undefined;
    let script = document.getElementById('eva-google-identity-services');
    let handleLoad;
    let handleError;
    const renderGoogleButton = () => {
      if (!window.google?.accounts?.id || !googleButtonRef.current) return;
      window.google.accounts.id.initialize({
        client_id: GOOGLE_CLIENT_ID,
        callback: async (credentialResponse) => {
          if (!credentialResponse?.credential) {
            setAuthError('Google sign-in did not return an account credential.');
            return;
          }
          setAuthError('');
          setAuthBusy(true);
          try {
            const response = await fetch(`${API_BASE}/v1/auth/google`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ credential: credentialResponse.credential }),
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || 'Google sign-in failed.');
            localStorage.setItem(AUTH_TOKEN_KEY, result.token);
            setAccount(result.user);
            setAccountDialogOpen(false);
            setAccountPassword('');
            setAccountReady(false);
            setAccountToken(result.token);
          } catch (googleError) {
            setAuthError(googleError.message || 'Google sign-in failed.');
          } finally {
            setAuthBusy(false);
          }
        },
      });
      window.google.accounts.id.renderButton(googleButtonRef.current, {
        theme: 'outline', size: 'large', shape: 'pill', text: 'continue_with', logo_alignment: 'left',
        width: Math.min(400, Math.floor(googleButtonRef.current.clientWidth || 360)),
      });
    };
    if (window.google?.accounts?.id) {
      renderGoogleButton();
      return undefined;
    }
    if (!script) {
      script = document.createElement('script');
      script.id = 'eva-google-identity-services';
      script.src = 'https://accounts.google.com/gsi/client';
      script.async = true;
      script.defer = true;
      document.head.appendChild(script);
    }
    handleLoad = renderGoogleButton;
    handleError = () => setAuthError('Could not load Google sign-in. Check your internet connection.');
    script.addEventListener('load', handleLoad, { once: true });
    script.addEventListener('error', handleError, { once: true });
    return () => {
      script?.removeEventListener('load', handleLoad);
      script?.removeEventListener('error', handleError);
    };
  }, [accountDialogOpen, account]);

  function signOut() {
    setAccountDialogOpen(false);
    setSignOutError('');
    setSignOutConfirmOpen(true);
  }

  async function confirmSignOut() {
    setSignOutBusy(true);
    setSignOutError('');
    try {
      if (accountToken && account) {
        const headers = { Authorization: `Bearer ${accountToken}`, 'Content-Type': 'application/json' };
        const saved = await Promise.all(conversations
          .filter((conversation) => conversation.messages?.length)
          .map((conversation) => fetch(`${API_BASE}/v1/conversations/${encodeURIComponent(conversation.id)}`, {
            method: 'PUT', headers, body: JSON.stringify(conversation),
          })));
        if (saved.some((response) => !response.ok)) throw new Error('The server rejected a chat save.');
      }
      stopVoiceConversation({ discard: true });
      localStorage.removeItem(AUTH_TOKEN_KEY);
      const guestConversation = newConversation();
      try {
        localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify([guestConversation]));
        localStorage.setItem(ACTIVE_CONVERSATION_KEY, guestConversation.id);
        localStorage.removeItem(STORAGE_KEY);
      } catch {
        // The server copy has already been saved; keep signing out if browser storage is unavailable.
      }
      setAccountToken('');
      setAccount(null);
      setAccountEmail('');
      setAccountPassword('');
      setAccountReady(true);
      setConversations([guestConversation]);
      setActiveConversationId(guestConversation.id);
      setMessages([]);
      setShowEmptyDraftInSpace(false);
      setSignOutConfirmOpen(false);
    } catch {
      setSignOutError('EVA could not save your latest chats. Check that the server is running, then try again. You are still signed in.');
    } finally {
      setSignOutBusy(false);
    }
  }

  async function refreshKnowledgeDocuments() {
    try {
      const response = await fetch(`${API_BASE}/v1/documents`);
      if (!response.ok) throw new Error('Could not load knowledge files.');
      setKnowledgeDocuments((await response.json()).documents || []);
      setDocumentError('');
    } catch (loadError) { setDocumentError(loadError.message); }
  }

  async function addKnowledgeDocument(file) {
    if (!file) return;
    setDocumentBusy(true);
    setDocumentError('');
    try {
      const form = new FormData(); form.append('file', file);
      const response = await fetch(`${API_BASE}/v1/documents`, { method: 'POST', body: form });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || 'Could not add that file.');
      await refreshKnowledgeDocuments();
    } catch (uploadError) { setDocumentError(uploadError.message); }
    finally { setDocumentBusy(false); if (knowledgeInputRef.current) knowledgeInputRef.current.value = ''; }
  }

  async function removeKnowledgeDocument(id) {
    try {
      const response = await fetch(`${API_BASE}/v1/documents/${encodeURIComponent(id)}`, { method: 'DELETE' });
      if (!response.ok) throw new Error('Could not delete that file.');
      setKnowledgeDocuments((items) => items.filter((item) => item.id !== id));
    } catch (deleteError) { setDocumentError(deleteError.message); }
  }

  async function sendMessage(text = draft, { fromVoice = false, audioFile = null } = {}) {
    const message = text.trim();
    if (!message || busyRef.current || (!fromVoice && (listening || recording))) return;
    const textAttachment = !fromVoice && !audioFile ? pendingAttachment : null;
    const prompt = textAttachment
      ? `${message}\n\nAttached file: ${textAttachment.name}\n\n${textAttachment.content}`
      : message;
    busyRef.current = true;
    const requestStartedAt = performance.now();
    const priorMessages = messagesRef.current;
    const history = [];
    for (let index = 0; index < priorMessages.length - 1; index += 2) {
      if (priorMessages[index]?.role === 'user' && priorMessages[index + 1]?.role === 'assistant') {
        history.push({ user: priorMessages[index].content, assistant: priorMessages[index + 1].content });
      }
    }
    const userMessage = {
      id: crypto.randomUUID(), role: 'user', content: message,
      ...(textAttachment ? { attachment: { name: textAttachment.name, type: textAttachment.type } } : {}),
    };
    setMessages([...priorMessages, userMessage]);
    setDraft('');
    if (textAttachment) setPendingAttachment(null);
    setError('');
    setBusy(true);
    try {
      let response;
      if (audioFile) {
        const form = new FormData();
        form.append('message', message);
        form.append('history_json', JSON.stringify(history.slice(-10)));
        form.append('adapt_emotion', String(adaptEmotion));
        if (liveAffectScoresRef.current) {
          form.append('live_affect_json', JSON.stringify(liveAffectScoresRef.current));
        }
        form.append('audio', audioFile, audioFile.name || 'voice-turn.webm');
        response = await fetch(`${API_BASE}/v1/turn`, { method: 'POST', body: form });
      } else {
        response = await fetch(`${API_BASE}/v1/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: prompt, history: history.slice(-10), adapt_emotion: adaptEmotion }),
        });
      }
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || 'EVA could not send a reply.');
      if (textAttachment) setPendingAttachment(null);
      setMessages((current) => [...current, {
        id: crypto.randomUUID(), role: 'assistant', content: result.response, affect: result.affect,
        latencyMs: Math.round(performance.now() - requestStartedAt),
      }]);
      if (voiceModeRef.current) await speakAssistantResponse(result.response);
    } catch (requestError) {
      setError(requestError.message || 'Could not reach EVA. Check that the API is running.');
    } finally {
      busyRef.current = false;
      setBusy(false);
      liveAffectScoresRef.current = null;
      scheduleVoiceListening();
    }
  }

  async function handleMessageAttachment(file) {
    if (!file) return;
    const extension = file.name.split('.').pop()?.toLowerCase() || '';
    setError('');
    if (SUPPORTED_AUDIO_EXTENSIONS.has(extension)) {
      await sendAudio(file);
      return;
    }
    if (file.type.startsWith('image/')) {
      setError('Image attachments are not supported yet: EVA’s current language model cannot read image contents. Attachments currently support audio and text/code files.');
      return;
    }
    if (!SUPPORTED_TEXT_EXTENSIONS.has(extension)) {
      setError(`Unsupported file type “.${extension || 'unknown'}”. Attach audio (${[...SUPPORTED_AUDIO_EXTENSIONS].join(', ')}) or text/code files (${[...SUPPORTED_TEXT_EXTENSIONS].join(', ')}).`);
      return;
    }
    if (file.size > MAX_CHAT_ATTACHMENT_BYTES) {
      setError('Text and code attachments must be 32 KB or smaller so EVA can fit them in its prompt.');
      return;
    }
    try {
      const content = await file.text();
      if (!content.trim()) throw new Error('That file is empty.');
      setPendingAttachment({ name: file.name, type: file.type || 'text/plain', content });
      setDraft((current) => current.trim() ? current : 'Please help me with this file.');
    } catch (attachmentError) {
      setError(attachmentError.message || 'EVA could not read that text file.');
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }

  async function sendAudio(file) {
    if (!file || busyRef.current) return;
    busyRef.current = true;
    const requestStartedAt = performance.now();
    setError('');
    setBusy(true);
    try {
      const form = new FormData();
      form.append('audio', file, file.name || 'recording.webm');
      form.append('adapt_emotion', String(adaptEmotion));
      const history = [];
      for (let index = 0; index + 1 < messagesRef.current.length; index += 2) {
        if (messagesRef.current[index]?.role === 'user' && messagesRef.current[index + 1]?.role === 'assistant') history.push({ user: messagesRef.current[index].content, assistant: messagesRef.current[index + 1].content });
      }
      form.append('history_json', JSON.stringify(history.slice(-10)));
      const response = await fetch(`${API_BASE}/v1/voice`, { method: 'POST', body: form });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || 'EVA could not process that audio.');
      if (!result.success) throw new Error(result.message || 'No speech was detected. Try again.');
      if (voiceModeRef.current && isNewConversationPhrase(result.transcription)) {
        restartVoiceAfterNewChatRef.current = true;
        startNewChat();
        return;
      }
      const requestedConversation = findConversationFromVoicePhrase(result.transcription);
      if (voiceModeRef.current && requestedConversation) {
        restartVoiceAfterNewChatRef.current = true;
        openConversation(requestedConversation.id);
        return;
      }
      if (voiceModeRef.current && isEndConversationPhrase(result.transcription)) {
        setMessages((current) => [...current, { id: crypto.randomUUID(), role: 'user', content: result.transcription, voice: true }]);
        stopVoiceConversation({ discard: true, resumeWake: false });
        if (wakeEnabledRef.current) scheduleWakePhraseListening();
        return;
      }
      setMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), role: 'user', content: result.transcription, voice: true },
        { id: crypto.randomUUID(), role: 'assistant', content: result.response, affect: result.affect, latencyMs: Math.round(performance.now() - requestStartedAt) },
      ]);
      if (voiceModeRef.current) await speakAssistantResponse(result.response);
    } catch (requestError) {
      setError(requestError.message || 'Could not process the audio.');
    } finally {
      const restartVoice = restartVoiceAfterNewChatRef.current;
      restartVoiceAfterNewChatRef.current = false;
      busyRef.current = false;
      setBusy(false);
      if (restartVoice) window.setTimeout(() => startVoiceConversation(), 0);
      else scheduleVoiceListening();
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }

  async function startAffectStream(stream) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass || !window.AudioWorkletNode || !window.WebSocket) return false;

    let context;
    let socket;
    try {
      context = new AudioContextClass();
      socket = new WebSocket(`${AFFECT_SOCKET_BASE}/v1/affect/stream`);
    } catch {
      await context?.close().catch(() => {});
      return false;
    }
    socket.binaryType = 'arraybuffer';
    affectSocketRef.current = socket;
    affectAudioContextRef.current = context;
    liveAffectScoresRef.current = null;
    affectQueuedFramesRef.current = [];

    socket.onopen = () => {
      socket.send(JSON.stringify({ sample_rate: context.sampleRate }));
      affectQueuedFramesRef.current.forEach((frame) => socket.send(frame));
      affectQueuedFramesRef.current = [];
      setAffectStreamActive(true);
    };
    socket.onmessage = (event) => {
      try {
        const result = JSON.parse(event.data);
        if ((result.type === 'affect' || result.type === 'final') && result.scores) {
          liveAffectScoresRef.current = result.scores;
        }
        if (result.type === 'final') affectFinalResolveRef.current?.(result.scores || null);
        if (result.type === 'error') setAffectStreamActive(false);
      } catch {
        // Ignore malformed streaming updates; the completed audio is the fallback.
      }
    };
    socket.onerror = () => setAffectStreamActive(false);
    socket.onclose = () => setAffectStreamActive(false);

    try {
      await context.audioWorklet.addModule(new URL('./voice-audio-worklet.js', import.meta.url));
      const source = context.createMediaStreamSource(stream);
      const worklet = new AudioWorkletNode(context, 'eva-affect-audio');
      const mute = context.createGain();
      mute.gain.value = 0;
      affectSourceRef.current = source;
      affectWorkletRef.current = worklet;
      affectMuteRef.current = mute;
      worklet.port.onmessage = (event) => {
        if (event.data?.type === 'flushed') {
          affectFlushResolveRef.current?.();
          affectFlushResolveRef.current = null;
          return;
        }
        if (event.data?.type !== 'audio' || !event.data.samples) return;
        const frame = event.data.samples.buffer;
        // Keep the browser's queue bounded if CPU inference falls behind.
        if (socket.readyState === WebSocket.OPEN && socket.bufferedAmount < 512_000) socket.send(frame);
        else if (socket.readyState === WebSocket.CONNECTING) {
          affectQueuedFramesRef.current.push(frame);
          if (affectQueuedFramesRef.current.length > 8) affectQueuedFramesRef.current.shift();
        }
      };
      source.connect(worklet);
      worklet.connect(mute);
      mute.connect(context.destination);
      await context.resume();
      return true;
    } catch {
      socket.onopen = null;
      socket.onmessage = null;
      socket.onerror = null;
      socket.onclose = null;
      if (socket.readyState < WebSocket.CLOSING) socket.close();
      await context.close().catch(() => {});
      affectSocketRef.current = null;
      affectAudioContextRef.current = null;
      setAffectStreamActive(false);
      return false;
    }
  }

  async function finishAffectStream() {
    const worklet = affectWorkletRef.current;
    if (worklet) {
      await new Promise((resolve) => {
        const timeout = window.setTimeout(resolve, 350);
        affectFlushResolveRef.current = () => {
          window.clearTimeout(timeout);
          resolve();
        };
        worklet.port.postMessage({ type: 'flush' });
      });
      worklet.port.onmessage = null;
      worklet.disconnect();
      affectWorkletRef.current = null;
    }
    affectSourceRef.current?.disconnect();
    affectSourceRef.current = null;
    affectMuteRef.current?.disconnect();
    affectMuteRef.current = null;
    if (affectAudioContextRef.current) {
      await affectAudioContextRef.current.close().catch(() => {});
      affectAudioContextRef.current = null;
    }

    const socket = affectSocketRef.current;
    let finalScores = null;
    if (socket?.readyState === WebSocket.CONNECTING) {
      await Promise.race([
        new Promise((resolve) => socket.addEventListener('open', resolve, { once: true })),
        new Promise((resolve) => window.setTimeout(resolve, 500)),
      ]);
    }
    if (socket?.readyState === WebSocket.OPEN) {
      finalScores = await new Promise((resolve) => {
        const timeout = window.setTimeout(() => resolve(null), 4000);
        affectFinalResolveRef.current = (scores) => {
          window.clearTimeout(timeout);
          affectFinalResolveRef.current = null;
          resolve(scores);
        };
        socket.send(JSON.stringify({ type: 'finish' }));
      });
      affectFinalResolveRef.current = null;
    }
    if (socket && socket.readyState < WebSocket.CLOSING) socket.close();
    affectSocketRef.current = null;
    affectQueuedFramesRef.current = [];
    setAffectStreamActive(false);
    const scores = finalScores || liveAffectScoresRef.current;
    liveAffectScoresRef.current = scores;
    return scores;
  }

  async function beginVoiceTurnCapture() {
    if (!adaptEmotion || !navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) return;
    try {
      liveAffectScoresRef.current = null;
      if (!turnStreamRef.current || turnStreamRef.current.getTracks().every((track) => track.readyState === 'ended')) {
        turnStreamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
      }
      await startAffectStream(turnStreamRef.current);
      const preferred = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/mp4']
        .find((type) => MediaRecorder.isTypeSupported(type));
      const recorder = new MediaRecorder(turnStreamRef.current, preferred ? { mimeType: preferred } : undefined);
      turnRecorderRef.current = recorder;
      turnChunksRef.current = [];
      turnCapturePromiseRef.current = new Promise((resolve) => {
        recorder.ondataavailable = (event) => { if (event.data.size) turnChunksRef.current.push(event.data); };
        recorder.onstop = () => {
          const blob = new Blob(turnChunksRef.current, { type: recorder.mimeType || 'audio/webm' });
          const extension = blob.type.includes('ogg') ? 'ogg' : blob.type.includes('mp4') ? 'm4a' : 'webm';
          resolve(blob.size ? new File([blob], `voice-turn.${extension}`, { type: blob.type }) : null);
          if (turnRecorderRef.current === recorder) turnRecorderRef.current = null;
          turnCapturePromiseRef.current = null;
        };
      });
      recorder.start();
    } catch {
      await finishAffectStream();
      setError('Voice tone analysis is unavailable. EVA will use the transcript only.');
      turnStreamRef.current?.getTracks().forEach((track) => track.stop());
      turnStreamRef.current = null;
    }
  }

  async function finishVoiceTurnCapture({ discard = false, stopStream = false } = {}) {
    const recorder = turnRecorderRef.current;
    if (recorder?.state === 'recording') recorder.stop();
    let file = null;
    if (turnCapturePromiseRef.current) file = await turnCapturePromiseRef.current;
    const affectScores = await finishAffectStream();
    liveAffectScoresRef.current = discard ? null : affectScores;
    if (stopStream) {
      turnStreamRef.current?.getTracks().forEach((track) => track.stop());
      turnStreamRef.current = null;
    }
    return discard ? null : file;
  }

  async function startLiveDictation({ autoSend = false, bargeIn = false } = {}) {
    if (speechMode === 'whisper' && !bargeIn) {
      startAudioRecording();
      return;
    }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      if (bargeIn) {
        setError('This browser cannot listen for interruptions while EVA speaks. Use Chrome or Edge for voice interruption.');
        return;
      }
      setError('Live dictation is not supported in this browser. Recording will be sent when you stop.');
      startAudioRecording();
      return;
    }

    // Browser speech recognition and the always-on wake recognizer compete for
    // the same microphone/service in some browsers. Pause wake detection for
    // an explicit dictation turn and resume it when that turn ends.
    if (!autoSend) stopWakePhraseListener();
    try {
      if (autoSend && adaptEmotion && !bargeIn) await beginVoiceTurnCapture();
      if (autoSend && !voiceModeRef.current) {
        await finishVoiceTurnCapture({ discard: true, stopStream: true });
        return;
      }
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = navigator.language || 'en-US';
      speechBaseRef.current = autoSend ? '' : draft.trim();
      speechFinalRef.current = '';
      let heardSpeech = false;
      let recorderFallbackNeeded = false;
      recognition.onresult = (event) => {
        let finalText = '';
        let interimText = '';
        for (let index = 0; index < event.results.length; index += 1) {
          const result = event.results[index];
          if (result.isFinal) finalText += result[0].transcript;
          else interimText += result[0].transcript;
        }
        if (finalText || interimText) heardSpeech = true;
        const recognizedText = [finalText, interimText].map((part) => part.trim()).filter(Boolean).join(' ');
        const explicitVoiceCommand = isEndConversationPhrase(recognizedText) || isNewConversationPhrase(recognizedText);
        if (bargeInRef.current && isAssistantSpeechEcho(recognizedText) && !explicitVoiceCommand) {
          speechFinalRef.current = '';
          setDraft('');
          scheduleRecognitionSilenceStop();
          return;
        }
        if (bargeInRef.current && recognizedText) interruptAssistantSpeech();
        speechFinalRef.current = finalText.trim();
        scheduleRecognitionSilenceStop();
        setDraft([speechBaseRef.current, finalText, interimText]
          .map((part) => part.trim()).filter(Boolean).join(' '));
      };
      recognition.onstart = () => {
        clearSilenceTimer();
        silenceTimerRef.current = window.setTimeout(() => {
          if (!heardSpeech) recognition.stop();
        }, initialVoiceTimeoutMs);
      };
      recognition.onspeechstart = () => {
        heardSpeech = true;
        clearSilenceTimer();
      };
      recognition.onspeechend = scheduleRecognitionSilenceStop;
      recognition.onerror = (event) => {
        if (event.error === 'not-allowed') {
          stopVoiceConversation({ discard: true });
          setError('Microphone or speech access was blocked. Allow access in your browser and try again.');
        } else if (['no-speech', 'network', 'language-not-supported', 'service-not-allowed'].includes(event.error)) {
          // Browser recognition depends on the browser's speech service and
          // may fail even when microphone capture works. Fall back to a local
          // recording sent to EVA's Whisper endpoint when this turn finishes.
          recorderFallbackNeeded = true;
        } else if (event.error !== 'aborted' && event.error !== 'no-speech') {
          stopVoiceConversation({ discard: true });
          setError(`Live dictation stopped (${event.error}). You can try again or attach an audio file.`);
        } else {
          setListening(false);
        }
      };
      let autoSendHandled = false;
      recognition.onend = async () => {
        clearSilenceTimer();
        setListening(false);
        recognitionRef.current = null;
        const transcript = [speechBaseRef.current, speechFinalRef.current]
          .map((part) => part.trim()).filter(Boolean).join(' ');
        setDraft(transcript);
        if (bargeIn && !transcript) {
          if (voiceModeRef.current) {
            if (speechPlaybackRef.current) startLiveDictation({ autoSend: true, bargeIn: true });
            else scheduleVoiceListening();
          }
          return;
        }
        if (!transcript && recorderFallbackNeeded) {
          await finishVoiceTurnCapture({ discard: true, stopStream: true });
          setError('Browser speech recognition did not catch your words. Switching to microphone recording and Whisper.');
          startAudioRecording();
          return;
        }
        if (autoSend && !autoSendHandled) {
          autoSendHandled = true;
          if (voiceModeRef.current && isNewConversationPhrase(transcript)) {
            setDraft('');
            await finishVoiceTurnCapture({ discard: true, stopStream: true });
            startNewChat();
            startVoiceConversation();
            return;
          }
          const requestedConversation = findConversationFromVoicePhrase(transcript);
          if (voiceModeRef.current && requestedConversation) {
            setDraft('');
            await finishVoiceTurnCapture({ discard: true, stopStream: true });
            openConversation(requestedConversation.id);
            startVoiceConversation();
            return;
          }
          if (voiceModeRef.current && isEndConversationPhrase(transcript)) {
            setDraft('');
            await finishVoiceTurnCapture({ discard: true, stopStream: true });
            stopVoiceConversation({ discard: true, resumeWake: false });
            scheduleWakePhraseListening();
            return;
          }
          if (transcript.trim()) {
            const audioFile = await finishVoiceTurnCapture({ stopStream: !voiceModeRef.current });
            sendMessage(transcript, { fromVoice: true, audioFile });
          } else {
            await finishVoiceTurnCapture({ discard: true, stopStream: !voiceModeRef.current });
            if (voiceModeRef.current) {
              setError('I didn’t catch that. Switching to microphone recording and Whisper.');
              startAudioRecording();
            }
            else setError('I didn’t catch a question. Try speaking again.');
          }
          if (!voiceModeRef.current && wakeEnabledRef.current) scheduleWakePhraseListening();
        } else if (!voiceModeRef.current && wakeEnabledRef.current) {
          scheduleWakePhraseListening();
        }
      };
      recognition.start();
      recognitionRef.current = recognition;
      setError('');
      setListening(true);
    } catch {
      setError('Could not start live dictation. Check microphone permission and try again.');
      setListening(false);
      if (!autoSend && wakeEnabledRef.current) scheduleWakePhraseListening(1000);
    }
  }

  function scheduleRecognitionSilenceStop() {
    clearSilenceTimer();
    silenceTimerRef.current = window.setTimeout(() => {
      recognitionRef.current?.stop();
    }, silenceTimeoutMs);
  }

  function startVoiceConversation() {
    if (voiceModeRef.current || busy) return;
    stopWakePhraseListener();
    voiceModeRef.current = true;
    setVoiceMode(true);
    setError('');
    startLiveDictation({ autoSend: true });
  }

  function stopVoiceConversation({ discard = false, resumeWake = true } = {}) {
    voiceModeRef.current = false;
    setVoiceMode(false);
    if (restartListeningTimerRef.current !== null) {
      window.clearTimeout(restartListeningTimerRef.current);
      restartListeningTimerRef.current = null;
    }
    ttsAbortControllerRef.current?.abort();
    ttsAbortControllerRef.current = null;
    audioCancelRef.current?.();
    audioCancelRef.current = null;
    if (recognitionRef.current) {
      if (discard) {
        recognitionRef.current.onresult = null;
        recognitionRef.current.onerror = null;
        recognitionRef.current.onend = null;
        recognitionRef.current.abort();
        recognitionRef.current = null;
      } else {
        recognitionRef.current.stop();
      }
    }
    if (recorderRef.current?.state === 'recording') {
      if (discard) recorderRef.current.onstop = null;
      recorderRef.current.stop();
      if (discard) {
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    }
    if (discard) {
      finishVoiceTurnCapture({ discard: true, stopStream: true });
      stopSilenceMonitor();
      setListening(false);
      setRecording(false);
    }
    if (resumeWake && wakeEnabledRef.current && !recognitionRef.current) scheduleWakePhraseListening();
  }

  async function startAudioRecording() {
    if (recording) {
      recorderRef.current?.stop();
      setRecording(false);
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setError('This browser does not support microphone recording. You can attach an audio file instead.');
      return;
    }
    if (!voiceModeRef.current) stopWakePhraseListener();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const preferred = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/mp4']
        .find((type) => MediaRecorder.isTypeSupported(type));
      const recorder = new MediaRecorder(stream, preferred ? { mimeType: preferred } : undefined);
      recorderRef.current = recorder;
      chunksRef.current = [];
      recorder.ondataavailable = (event) => { if (event.data.size) chunksRef.current.push(event.data); };
      recorder.onstop = () => {
        stopSilenceMonitor();
        const audio = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' });
        const extension = audio.type.includes('ogg') ? 'ogg' : audio.type.includes('mp4') ? 'm4a' : 'webm';
        stream.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        if (audio.size) {
          sendAudio(new File([audio], `recording.${extension}`, { type: audio.type }))
            .finally(() => {
              if (!voiceModeRef.current && wakeEnabledRef.current) scheduleWakePhraseListening();
            });
        } else if (!voiceModeRef.current && wakeEnabledRef.current) {
          scheduleWakePhraseListening();
        }
      };
      recorder.start();
      setError('');
      setRecording(true);
      watchForAudioSilence(stream, () => {
        if (recorder.state === 'recording') recorder.stop();
        setRecording(false);
      });
    } catch {
      if (voiceModeRef.current) stopVoiceConversation({ discard: true });
      setError('Microphone access was blocked. Allow microphone access or attach an audio file.');
      if (wakeEnabledRef.current) scheduleWakePhraseListening(1000);
    }
  }

  function toggleMicrophone() {
    if (listening) {
      recognitionRef.current?.stop();
      return;
    }
    if (recording) {
      recorderRef.current?.stop();
      setRecording(false);
      return;
    }
    startLiveDictation();
  }

  wakeActionsRef.current = { startNewChat, startVoiceConversation, openConversation };
  const visibleConversations = conversations
    .map((conversation) => conversation.id === activeConversationId
      ? { ...conversation, messages }
      : conversation)
    .filter((conversation) => (
      conversation.messages.length > 0
      || (showEmptyDraftInSpace && conversation.id === activeConversationId)
    ))
    .sort((first, second) => second.updatedAt - first.updatedAt);
  const searchQuery = spaceSearch.trim().toLowerCase();
  const filteredConversations = visibleConversations.filter((conversation) => (
    !searchQuery
    || conversation.title.toLowerCase().includes(searchQuery)
    || conversation.messages.some((message) => message.content.toLowerCase().includes(searchQuery))
  ));

  function startSidebarResize(event) {
    if (event.button !== 0) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    sidebarDragRef.current = { pointerId: event.pointerId, startX: event.clientX, startWidth: sidebarWidth };
    setSidebarResizing(true);
  }

  function moveSidebarResize(event) {
    const drag = sidebarDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const width = Math.max(72, Math.min(320, drag.startWidth + event.clientX - drag.startX));
    setSidebarWidth(width);
    setSidebarCollapsed(width < 145);
  }

  function finishSidebarResize(event) {
    const drag = sidebarDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const width = Math.max(72, Math.min(320, drag.startWidth + event.clientX - drag.startX));
    if (width < 160) {
      setSidebarWidth(72);
      setSidebarCollapsed(true);
    } else {
      setSidebarWidth(Math.max(180, width));
      setSidebarCollapsed(false);
    }
    sidebarDragRef.current = null;
    setSidebarResizing(false);
  }

  return (
    <div className="app-shell">
      <aside
        className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''} ${sidebarCollapsed ? 'sidebar-collapsed' : ''} ${sidebarResizing ? 'sidebar-resizing' : ''}`}
        style={{ '--sidebar-width': `${sidebarWidth}px` }}
      >
        <div className="sidebar-top">
          <div className="sidebar-brand-row">
            <a className="brand" href="#top" onClick={(event) => { event.preventDefault(); startNewChat(); }} aria-label="Start a new conversation">
              <span className="brand-mark"><EvaTreeMark size={25} /></span>
              <span>EVA</span>
              <span className="brand-version">PERSONAL</span>
            </a>
            <button className="icon-button sidebar-toggle" onClick={() => {
              const nextCollapsed = !sidebarCollapsed;
              setSidebarCollapsed(nextCollapsed);
              setSidebarWidth(nextCollapsed ? 72 : 258);
            }} aria-label={sidebarCollapsed ? 'Show sidebar' : 'Hide sidebar'}>
              {sidebarCollapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
              <span className="button-tooltip" aria-hidden="true">{sidebarCollapsed ? 'Show sidebar' : 'Hide sidebar'}</span>
            </button>
          </div>
          <button className="icon-button sidebar-close" onClick={() => setSidebarOpen(false)} aria-label="Close menu"><X size={18} /></button>
          <button className="new-chat-button" onClick={startNewChat} aria-label="New conversation, Alt plus K">
            <MessageSquarePlus size={17} /><span>New conversation</span><span className="shortcut">Alt K</span>
            <span className="button-tooltip button-tooltip-above" aria-hidden="true">Start a new conversation · Alt + K</span>
          </button>
          <div className="sidebar-space-tools" ref={spaceSearchRef}>
            <div className="sidebar-space-heading">
              <div className="sidebar-label">YOUR SPACE</div>
              <button
                className="space-search-toggle"
                onClick={() => { setSpaceSearchOpen((open) => !open); setSpaceSearch(''); }}
                aria-label={spaceSearchOpen ? 'Close chat search' : 'Search chats'}
              ><Search size={15} /><span className="button-tooltip" aria-hidden="true">{spaceSearchOpen ? 'Close chat search' : 'Search chats'}</span></button>
            </div>
            {spaceSearchOpen && (
              <div className="space-search-box">
                <Search size={14} aria-hidden="true" />
                <input
                  autoFocus
                  aria-label="Search chats"
                  placeholder="Search chats or messages"
                  value={spaceSearch}
                  onChange={(event) => setSpaceSearch(event.target.value)}
                  onKeyDown={(event) => { if (event.key === 'Escape') { setSpaceSearch(''); setSpaceSearchOpen(false); } }}
                />
                {spaceSearch && <button onClick={() => setSpaceSearch('')} aria-label="Clear chat search"><X size={13} /></button>}
              </div>
            )}
          </div>
          <div className="conversation-items">
            {filteredConversations.map((conversation) => (
              <div
                className={`conversation-item ${conversation.id === activeConversationId ? 'conversation-item-active' : ''} ${renamingConversationId === conversation.id ? 'conversation-item-renaming' : ''}`}
                key={conversation.id}
              >
                {renamingConversationId === conversation.id ? (
                  <div className="conversation-rename-form" ref={renameFormRef}>
                    <input
                      ref={renameInputRef}
                      aria-label={`Rename ${conversation.title}`}
                      value={renameDraft}
                      onChange={(event) => setRenameDraft(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') saveConversationRename(conversation.id);
                        if (event.key === 'Escape') cancelRenameConversation();
                      }}
                    />
                    <button onClick={() => saveConversationRename(conversation.id)} aria-label="Save chat name" title="Save"><Check size={14} /></button>
                    <button onClick={cancelRenameConversation} aria-label="Cancel rename" title="Cancel"><X size={14} /></button>
                  </div>
                ) : (
                  <>
                    <button className="conversation-open" onClick={() => openConversation(conversation.id)} title={conversation.title}>
                      <MessageSquarePlus size={15} />
                      <span>{conversation.title}</span>
                    </button>
                    <button
                      className="conversation-rename-button"
                      onClick={() => beginRenameConversation(conversation)}
                      aria-label={`Rename ${conversation.title}`}
                    ><Pencil size={13} /><span className="button-tooltip button-tooltip-above" aria-hidden="true">Rename chat</span></button>
                    <button
                      className="conversation-delete"
                      onClick={() => deleteConversation(conversation.id)}
                      aria-label={`Delete ${conversation.title}`}
                    ><Trash2 size={14} /><span className="button-tooltip button-tooltip-above button-tooltip-danger" aria-hidden="true">Delete chat</span></button>
                  </>
                )}
              </div>
            ))}
            {searchQuery && filteredConversations.length === 0 && <div className="space-search-empty">No matching chats</div>}
          </div>
        </div>
        <div className="sidebar-bottom">
          <div className="profile-action-row">
            <button className="profile-row" onClick={() => {
              setAccountMode('login');
              if (!account) { setAccountEmail(''); setAccountPassword(''); setAuthError(''); setAuthNotice(''); }
              setAccountDialogOpen(true);
            }}>
              <span className="avatar user-avatar">{account?.email?.[0]?.toUpperCase() || 'A'}</span>
              <span className="profile-name">{account ? accountDisplayName(account.email) : 'Sign in to sync chats'}</span>
              {!account && <LogIn size={16} />}
            </button>
            {account && <button type="button" className="account-signout" onClick={signOut}>Sign out</button>}
          </div>
        </div>
        <div
          className="sidebar-resize-handle"
          role="separator"
          aria-label="Drag to resize or collapse sidebar"
          aria-orientation="vertical"
          aria-valuemin={72}
          aria-valuemax={320}
          aria-valuenow={Math.round(sidebarWidth)}
          tabIndex={0}
          onKeyDown={(event) => {
            if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
            event.preventDefault();
            const width = Math.max(72, Math.min(320, sidebarWidth + (event.key === 'ArrowLeft' ? -16 : 16)));
            setSidebarWidth(width < 160 ? 72 : width);
            setSidebarCollapsed(width < 160);
          }}
          onPointerDown={startSidebarResize}
          onPointerMove={moveSidebarResize}
          onPointerUp={finishSidebarResize}
          onPointerCancel={finishSidebarResize}
        />
      </aside>
      {sidebarOpen && <button className="scrim" onClick={() => setSidebarOpen(false)} aria-label="Close navigation" />}

      <main className="main-panel" id="top">
        <header className="topbar">
          <button className="icon-button mobile-menu" onClick={() => setSidebarOpen(true)} aria-label="Open menu"><Menu size={19} /></button>
          <div className="topbar-actions">
            <button className="topbar-settings" onClick={() => setSettingsOpen((open) => !open)} aria-label="Voice and reply settings" aria-expanded={settingsOpen}><Settings2 size={17} /><span className="button-tooltip" aria-hidden="true">Voice and reply settings</span></button>
            <button
              ref={aboutButtonRef}
              className="topbar-help"
              type="button"
              aria-label="About EVA"
              aria-haspopup="dialog"
              aria-expanded={aboutOpen}
              onClick={() => setAboutOpen(true)}
            >
              <CircleHelp size={18} />
              <span className="button-tooltip" role="tooltip">About EVA</span>
            </button>
            {settingsOpen && (
              <div className="settings-popover" role="dialog" aria-label="Voice and reply settings">
                <strong>Preferences</strong>
                <button className="preference-row" role="switch" aria-checked={adaptEmotion} onClick={() => setAdaptEmotion((enabled) => !enabled)}>
                  <span><b>Adapt reply style</b><small>Use uncertain text and voice cues to adjust tone.</small></span>
                  <i className={adaptEmotion ? 'preference-switch preference-switch-on' : 'preference-switch'} />
                </button>
                <button className="preference-row" role="switch" aria-checked={wakeEnabled} onClick={() => setWakeEnabled((enabled) => !enabled)}>
                  <span><b>Wake phrase listening</b><small>Say “EVA” anywhere to start, or say “open a new chat” to create one and start listening.</small></span>
                  <i className={wakeEnabled ? 'preference-switch preference-switch-on' : 'preference-switch'} />
                </button>
                <button className="preference-row" role="switch" aria-checked={speechMode === 'whisper'} onClick={() => setSpeechMode((mode) => mode === 'whisper' ? 'browser' : 'whisper')}>
                  <span><b>Local Whisper speech input</b><small>{speechMode === 'whisper' ? 'Records each turn, then transcribes after you pause; words do not appear live.' : 'Browser mode displays words while you speak; it needs browser speech-recognition support.'}</small></span>
                  <i className={speechMode === 'whisper' ? 'preference-switch preference-switch-on' : 'preference-switch'} />
                </button>
                <button className="preference-row" role="switch" aria-checked={ttsMode === 'device'} onClick={() => setTtsMode((mode) => mode === 'device' ? 'edge' : 'device')}>
                  <span><b>Use device voice</b><small>{ttsMode === 'device' ? 'Uses an installed browser/device voice; voice quality and gender depend on the device.' : 'Uses the feminine Aria Neural voice through Edge TTS (internet required).'}</small></span>
                  <i className={ttsMode === 'device' ? 'preference-switch preference-switch-on' : 'preference-switch'} />
                </button>
                <div className="knowledge-section">
                  <div className="knowledge-heading"><span><b>Knowledge files</b><small>Ground answers in your local .txt and .md files.</small></span>
                    <button type="button" className="knowledge-add" disabled={documentBusy} onClick={() => knowledgeInputRef.current?.click()}><Paperclip size={13} /> Add</button>
                  </div>
                  <input ref={knowledgeInputRef} className="visually-hidden" type="file" accept=".txt,.md,text/plain,text/markdown" onChange={(event) => addKnowledgeDocument(event.target.files?.[0])} />
                  {knowledgeDocuments.length ? <div className="knowledge-list">{knowledgeDocuments.map((doc) => <div className="knowledge-item" key={doc.id}><span title={doc.name}>{doc.name}</span><button type="button" onClick={() => removeKnowledgeDocument(doc.id)} aria-label={`Remove ${doc.name}`}><Trash2 size={13} /></button></div>)}</div> : <p className="knowledge-empty">{documentBusy ? 'Adding file…' : 'No files added yet.'}</p>}
                  {documentError && <p className="knowledge-error">{documentError}</p>}
                </div>
              </div>
            )}
          </div>
        </header>

        <section className={`conversation ${messages.length ? 'has-messages' : 'empty-conversation'}`}>
          {messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-mark"><EvaTreeMark size={39} /></div>
              <div className="eyebrow"><Sparkles size={13} /> YOUR SPACE TO TALK</div>
              <h1>Let’s talk it through.</h1>
              <p>Start a hands-free conversation with EVA. Speak naturally and stop whenever you’re ready.</p>
              <div className={`voice-panel ${voiceMode ? 'voice-panel-active' : ''}`}>
                <div className="voice-panel-heading">
                  <span>VOICE CONVERSATION</span>
                  <span className="voice-panel-mode"><i />{voiceMode ? 'LIVE' : 'HANDS-FREE'}</span>
                </div>
                <button
                  className={`voice-start ${voiceMode ? 'voice-start-active' : ''}`}
                  onClick={() => (voiceMode ? stopVoiceConversation() : startVoiceConversation())}
                  disabled={busy && !voiceMode}
                >
                  <span className="voice-start-icon">{voiceMode ? <Square size={21} fill="currentColor" /> : <Mic size={24} />}</span>
                  <span className="voice-start-copy">
                    <strong>{voiceMode ? 'Stop voice conversation' : 'Start a voice conversation'}</strong>
                    <small>{busy && voiceMode ? 'Speak anytime to interrupt, or say “end conversation”' : voiceMode && (listening || recording) ? 'Listening for your next turn' : voiceMode ? 'Voice conversation is active' : recording ? 'Recording a voice message; pause to transcribe' : listening ? 'Dictating live; pause to finish' : 'Talk with EVA hands-free until you stop'}</small>
                  </span>
                  <span className="voice-start-hint">{voiceMode ? 'STOP' : 'START'}</span>
                </button>
                <div className="voice-panel-footnote">Speak naturally <span>·</span> Say “new conversation” to start a fresh chat or “end conversation” to stop</div>
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message) => (
                <article className={`message message-${message.role}`} key={message.id}>
                  {message.role === 'assistant' ? <span className="avatar eva-avatar"><EvaTreeMark size={19} /></span> : null}
                  <div className="message-body">
                    {message.role === 'assistant' && <div className="message-author">EVA</div>}
                    <div className="message-content">
                      {message.role === 'assistant'
                        ? <AssistantMessageContent content={message.content} />
                        : message.content}
                    </div>
                    {message.attachment && <span className="message-attachment-label"><Paperclip size={12} />{message.attachment.name}</span>}
                    {message.voice && <span className="voice-label"><AudioLines size={12} /> Voice message</span>}
                    {message.role === 'assistant' && message.affect?.adapted && (
                      <div className="affect-note">
                        Emotion-aware reply: {responseStyleNames[message.affect.response_style] || 'personalized'}
                      </div>
                    )}
                    {message.role === 'assistant' && message.affect && (
                      <>
                        <div className="reply-feedback">
                          <span>{affectFeedback[message.id] ? 'Thanks for the feedback' : 'Was this reply useful?'}</span>
                          <button
                            className={affectFeedback[message.id]?.rating === 'helpful' ? 'reply-feedback-selected' : ''}
                            onClick={() => rateReply(message, 'helpful')}
                            aria-label="This reply was useful"
                            title="Useful"
                          ><ThumbsUp size={14} /></button>
                          <button
                            className={affectFeedback[message.id]?.rating === 'not_helpful' ? 'reply-feedback-selected' : ''}
                            onClick={() => rateReply(message, 'not_helpful')}
                            aria-label="This reply was not useful"
                            title="Not useful"
                          ><ThumbsDown size={14} /></button>
                        </div>
                        {affectFeedback[message.id]?.rating === 'not_helpful' && (
                          <div className="reply-feedback-reasons">
                            {[
                              ['too_many_questions', 'Too many questions'],
                              ['tone_mismatch', 'Tone felt wrong'],
                              ['too_long', 'Too long'],
                              ['did_not_help', 'Didn’t solve it'],
                            ].map(([value, label]) => (
                              <button
                                key={value}
                                className={affectFeedback[message.id]?.reason === value ? 'reply-feedback-reason-selected' : ''}
                                onClick={() => rateReplyReason(message.id, value)}
                              >{label}</button>
                            ))}
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </article>
              ))}
              {busy && <article className="message message-assistant"><span className="avatar eva-avatar"><EvaTreeMark size={19} /></span><div className="typing-indicator" aria-label="EVA is thinking"><i /><i /><i /></div></article>}
              <div ref={messagesEndRef} />
            </div>
          )}
        </section>

        {voiceMode && messages.length > 0 && (
          <div className="voice-session-strip" role="status">
          <span className="voice-session-indicator"><i />{busy ? 'EVA is speaking · speak to interrupt' : listening || recording ? 'Listening for your next turn' : 'Voice conversation active'}</span>
            {affectStreamActive && <span className="voice-affect-status"><AudioLines size={13} />Reading voice cues live</span>}
            <button onClick={() => stopVoiceConversation()}><Square size={12} fill="currentColor" /> Stop conversation</button>
          </div>
        )}

        <div className="composer-area">
          {error && <div className="error-banner" role="alert"><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss error"><X size={15} /></button></div>}
          <form className={`composer ${recording ? 'is-recording' : ''} ${listening ? 'is-listening' : ''}`} onSubmit={(event) => { event.preventDefault(); sendMessage(); }}>
            {pendingAttachment && (
              <div className="pending-attachment">
                <Paperclip size={13} />
                <span title={pendingAttachment.name}>{pendingAttachment.name}</span>
                <button type="button" onClick={() => setPendingAttachment(null)} aria-label="Remove attachment"><X size={13} /></button>
              </div>
            )}
            <textarea
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendMessage(); }
              }}
              placeholder={listening ? 'Listening… your words will appear here' : recording ? 'Recording… tap the stop button when you’re done' : 'Message EVA…'}
              rows={1}
              disabled={recording}
              aria-label="Message EVA"
            />
            <div className="composer-actions">
              <div className="attach-actions">
                <button type="button" className="composer-icon" onClick={() => fileInputRef.current?.click()} disabled={busy || recording} aria-label="Attach audio, text, or code file"><Paperclip size={17} /><span className="button-tooltip button-tooltip-above attachment-tooltip" aria-hidden="true">Audio: .wav, .mp3, .m4a, .ogg, .flac, .webm<br />Text/code: .txt, .md, .csv, .json, common source files<br />Text/code files up to 32 KB. Images, PDFs, and Office files need additional model support.</span></button>
                <button type="button" className={`composer-icon mic-button ${recording ? 'recording' : ''} ${listening ? 'listening' : ''}`} onClick={toggleMicrophone} disabled={busy} aria-label={recording || listening ? 'Stop microphone' : 'Start live dictation'}>
                  {recording || listening ? <Square size={14} fill="currentColor" /> : <Mic size={17} />}
                  <span className="button-tooltip button-tooltip-above" aria-hidden="true">{recording || listening ? 'Stop microphone' : 'Speak to EVA'}</span>
                </button>
              </div>
              <button type="submit" className="send-button" disabled={busy || !draft.trim() || recording || listening} aria-label="Send message"><Send size={16} /></button>
            </div>
          </form>
          <div className="composer-footnote">{listening ? <span className="live-status"><i /> Listening live · pauses end each turn</span> : <span><Check size={12} /> {account ? 'Chats sync to your account' : 'Chat history saved in this browser'}</span>}<span>Enter to send · Shift + Enter for a new line</span></div>
        </div>
      </main>
      <input ref={fileInputRef} className="visually-hidden" type="file" accept=".wav,.mp3,.m4a,.ogg,.flac,.webm,.txt,.md,.csv,.json,.py,.js,.jsx,.ts,.tsx,.html,.css,.yaml,.yml,.xml,.log,.sh,.sql,.java,.c,.cpp,.h,.go,.rs,.php,.rb,.toml,.ini" onChange={(event) => handleMessageAttachment(event.target.files?.[0])} />
      {accountDialogOpen && <div className="modal-scrim" onMouseDown={(event) => { if (event.target === event.currentTarget) setAccountDialogOpen(false); }}>
        <form className="account-dialog" autoComplete="off" onSubmit={submitAccount}>
          <button type="button" className="dialog-close" onClick={() => setAccountDialogOpen(false)} aria-label="Close"><X size={17} /></button>
          <div className="account-dialog-heading">
            <span className="account-dialog-mark"><EvaTreeMark size={28} /></span>
            <span className="account-dialog-kicker">YOUR PERSONAL VOICE ASSISTANT</span>
          </div>
          <h2>{account ? 'Your EVA account' : accountMode === 'register' ? 'Create your EVA account' : 'Sign in to EVA'}</h2>
          {authError && <div className="error-banner" role="alert">{authError}</div>}
          {authNotice && <div className="auth-success" role="status">{authNotice}</div>}
          {account ? <><p>Signed in as {account.email}. Your conversations sync with this EVA server.</p><button type="button" className="account-primary" onClick={signOut}>Sign out</button></> : <>
            <p className="account-dialog-intro">Keep your conversations synced and pick them up on another device using the same EVA server.</p>
            <label>Email<input name="eva-account-email" autoComplete="off" type="email" value={accountEmail} onChange={(event) => setAccountEmail(event.target.value)} required /><small>Use your Gmail or Yahoo address.</small></label>
            <label>Password<input autoComplete={accountMode === 'register' ? 'new-password' : 'current-password'} type="password" inputMode={accountMode === 'register' ? 'numeric' : undefined} pattern={accountMode === 'register' ? '[0-9]{6}' : undefined} minLength={accountMode === 'register' ? 6 : undefined} maxLength={accountMode === 'register' ? 6 : 1024} value={accountPassword} onChange={(event) => setAccountPassword(event.target.value)} required />{accountMode === 'register' && <small>Use exactly 6 digits.</small>}</label>
            <button type="submit" className="account-primary" disabled={authBusy}>{authBusy ? 'Please wait…' : accountMode === 'register' ? 'Create account' : 'Sign in'}</button>
            <div className="auth-divider"><span>or continue with</span></div>
            {GOOGLE_CLIENT_ID
              ? <div className="google-signin-shell"><div className="google-signin-button" ref={googleButtonRef} /></div>
              : <button type="button" className="google-signin-fallback" onClick={() => setAuthError('Google sign-in needs an OAuth client ID. Follow the Google setup steps in README.md, then restart the frontend and backend.')}><span className="google-g-mark">G</span>Continue with Google</button>}
            {!GOOGLE_CLIENT_ID && <p className="google-auth-note">Google sign-in is visible, but requires an OAuth client ID to authenticate.</p>}
            <button type="button" className="account-mode-switch" onClick={() => { setAuthError(''); setAuthNotice(''); setAccountMode((mode) => mode === 'register' ? 'login' : 'register'); }}>{accountMode === 'register' ? 'Already have an account? Sign in' : 'New to EVA? Create an account'}</button>
          </>}
        </form>
      </div>}
      {signOutConfirmOpen && account && (
        <div className="modal-scrim logout-scrim" onMouseDown={(event) => { if (!signOutBusy && event.target === event.currentTarget) setSignOutConfirmOpen(false); }}>
          <section className="logout-dialog" role="dialog" aria-modal="true" aria-labelledby="logout-title" aria-describedby="logout-description">
            <h2 id="logout-title">Log out of EVA?</h2>
            <div className="logout-account">
              <span className="logout-avatar" aria-hidden="true">{(account.email?.split('@')[0]?.replace(/[._-]+/g, ' ').split(' ').map((part) => part[0]).join('').slice(0, 2) || 'E').toUpperCase()}</span>
              <span className="logout-account-copy">
                <strong>{account.email?.split('@')[0]?.replace(/[._-]+/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase()) || 'EVA account'}</strong>
                <small>{account.email}</small>
              </span>
            </div>
            <p id="logout-description">You’ll need to log in again to access your synced chats.</p>
            {signOutError && <div className="error-banner logout-error" role="alert">{signOutError}</div>}
            <button ref={signOutConfirmRef} type="button" className="logout-confirm-button" onClick={confirmSignOut} disabled={signOutBusy}>{signOutBusy ? 'Saving chats…' : 'Log out'}</button>
            <button type="button" className="logout-cancel-button" onClick={() => setSignOutConfirmOpen(false)} disabled={signOutBusy}>Cancel</button>
          </section>
        </div>
      )}
      {aboutOpen && (
        <div
          className="modal-scrim about-scrim"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              setAboutOpen(false);
              aboutButtonRef.current?.focus();
            }
          }}
        >
          <section className="about-dialog" role="dialog" aria-modal="true" aria-labelledby="about-eva-title" aria-describedby="about-eva-description">
            <button
              ref={aboutCloseRef}
              type="button"
              className="dialog-close"
              onClick={() => {
                setAboutOpen(false);
                aboutButtonRef.current?.focus();
              }}
              aria-label="Close About EVA"
            ><X size={17} /></button>
            <div className="about-dialog-heading">
              <span className="about-dialog-mark"><EvaTreeMark size={40} /></span>
              <span className="about-dialog-kicker">A LITTLE ABOUT EVA</span>
            </div>
            <h2 id="about-eva-title">Meet EVA</h2>
            <p className="about-dialog-intro" id="about-eva-description">
              EVA is your voice-first personal assistant. Ask by voice or text, and have a natural conversation at your own pace.
            </p>
            <div className="about-feature-list">
              <article className="about-feature">
                <span className="about-feature-icon"><AudioLines size={17} /></span>
                <span><strong>Speak or type</strong><small>Move between hands-free voice and regular chat.</small></span>
              </article>
              <article className="about-feature">
                <span className="about-feature-icon"><Sparkles size={17} /></span>
                <span><strong>Emotion-aware replies</strong><small>EVA estimates emotional cues and can adjust its tone to respond more directly, warmly, or encouragingly.</small></span>
              </article>
              <article className="about-feature">
                <span className="about-feature-icon"><MessageSquarePlus size={17} /></span>
                <span><strong>Your conversations</strong><small>Keep chats together and optionally add knowledge files for context.</small></span>
              </article>
            </div>
            <p className="about-dialog-note">Emotion cues are estimates and can be wrong. EVA is here to support you, not diagnose how you feel.</p>
          </section>
        </div>
      )}
    </div>
  );
}
