import React, { useEffect, useRef, useState } from 'react';
import {
  AudioLines, AudioWaveform, Check, ChevronDown, CircleHelp,
  Menu, MessageSquarePlus, Mic, PanelLeftClose, PanelLeftOpen,
  Paperclip, Send, Settings2, Sparkles,
  Square, X,
} from 'lucide-react';

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');
const STORAGE_KEY = 'eva-chat-messages-v1';
const CONVERSATIONS_KEY = 'eva-conversations-v1';
const ACTIVE_CONVERSATION_KEY = 'eva-active-conversation-v1';

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
const starters = [
  { title: 'Help me think', text: 'Help me think through a decision I’m facing.' },
  { title: 'Explain something', text: 'Explain a tricky idea in simple terms.' },
  { title: 'Make a plan', text: 'Help me make a practical plan for this week.' },
  { title: 'Just talk', text: 'I’d like to talk something through.' },
];

export default function App() {
  const [initialConversations] = useState(readSavedConversations);
  const [conversations, setConversations] = useState(initialConversations);
  const [activeConversationId, setActiveConversationId] = useState(() => readActiveConversationId(initialConversations));
  const [messages, setMessages] = useState(() => (
    initialConversations.find((conversation) => conversation.id === readActiveConversationId(initialConversations))
      || initialConversations[0]
  ).messages);
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [recording, setRecording] = useState(false);
  const [listening, setListening] = useState(false);
  const [voiceMode, setVoiceMode] = useState(false);
  const [error, setError] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
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
  const silenceTimeoutMs = 2000;

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
    if (!voiceModeRef.current) return;
    if (restartListeningTimerRef.current !== null) {
      window.clearTimeout(restartListeningTimerRef.current);
    }
    restartListeningTimerRef.current = window.setTimeout(() => {
      restartListeningTimerRef.current = null;
      if (voiceModeRef.current) startLiveDictation({ autoSend: true });
    }, 350);
  }

  async function speakAssistantResponse(text) {
    if (speechPlaybackRef.current) return speechPlaybackRef.current;
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
    const checkLevel = () => {
      analyser.getByteTimeDomainData(samples);
      let energy = 0;
      for (const sample of samples) {
        const amplitude = (sample - 128) / 128;
        energy += amplitude * amplitude;
      }
      const rms = Math.sqrt(energy / samples.length);
      if (rms > 0.025) lastVoiceAt = performance.now();
      if (performance.now() - lastVoiceAt >= silenceTimeoutMs) {
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

  useEffect(() => () => {
    voiceModeRef.current = false;
    if (restartListeningTimerRef.current !== null) window.clearTimeout(restartListeningTimerRef.current);
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
  }, []);

  function startNewChat() {
    stopVoiceConversation({ discard: true });
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
    const conversation = newConversation();
    setConversations((current) => [
      conversation,
      ...current.filter((item) => item.id !== activeConversationId || messages.length > 0),
    ]);
    setActiveConversationId(conversation.id);
    setMessages([]);
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
    setDraft('');
    setError('');
    setSidebarOpen(false);
  }

  async function sendMessage(text = draft, { fromVoice = false } = {}) {
    const message = text.trim();
    if (!message || busyRef.current || (!fromVoice && (listening || recording))) return;
    busyRef.current = true;
    const priorMessages = messagesRef.current;
    const history = [];
    for (let index = 0; index < priorMessages.length - 1; index += 2) {
      if (priorMessages[index]?.role === 'user' && priorMessages[index + 1]?.role === 'assistant') {
        history.push({ user: priorMessages[index].content, assistant: priorMessages[index + 1].content });
      }
    }
    const userMessage = { id: crypto.randomUUID(), role: 'user', content: message };
    setMessages([...priorMessages, userMessage]);
    setDraft('');
    setError('');
    setBusy(true);
    try {
      const response = await fetch(`${API_BASE}/v1/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, history: history.slice(-10) }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || 'EVA could not send a reply.');
      setMessages((current) => [...current, {
        id: crypto.randomUUID(), role: 'assistant', content: result.response,
      }]);
      if (voiceModeRef.current) await speakAssistantResponse(result.response);
    } catch (requestError) {
      setError(requestError.message || 'Could not reach EVA. Check that the API is running.');
    } finally {
      busyRef.current = false;
      setBusy(false);
      scheduleVoiceListening();
    }
  }

  async function sendAudio(file) {
    if (!file || busyRef.current) return;
    busyRef.current = true;
    setError('');
    setBusy(true);
    try {
      const form = new FormData();
      form.append('audio', file, file.name || 'recording.webm');
      const response = await fetch(`${API_BASE}/v1/voice`, { method: 'POST', body: form });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || 'EVA could not process that audio.');
      if (!result.success) throw new Error(result.message || 'No speech was detected. Try again.');
      setMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), role: 'user', content: result.transcription, voice: true },
        { id: crypto.randomUUID(), role: 'assistant', content: result.response },
      ]);
      if (voiceModeRef.current) await speakAssistantResponse(result.response);
    } catch (requestError) {
      setError(requestError.message || 'Could not process the audio.');
    } finally {
      busyRef.current = false;
      setBusy(false);
      scheduleVoiceListening();
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }

  function startLiveDictation({ autoSend = false } = {}) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setError('Live dictation is not supported in this browser. Recording will be sent when you stop.');
      startAudioRecording();
      return;
    }

    try {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = navigator.language || 'en-US';
      speechBaseRef.current = autoSend ? '' : draft.trim();
      speechFinalRef.current = '';
      recognition.onresult = (event) => {
        let finalText = '';
        let interimText = '';
        for (let index = 0; index < event.results.length; index += 1) {
          const result = event.results[index];
          if (result.isFinal) finalText += result[0].transcript;
          else interimText += result[0].transcript;
        }
        speechFinalRef.current = finalText.trim();
        scheduleRecognitionSilenceStop();
        setDraft([speechBaseRef.current, finalText, interimText]
          .map((part) => part.trim()).filter(Boolean).join(' '));
      };
      recognition.onstart = () => {
        if (!voiceModeRef.current) scheduleRecognitionSilenceStop();
      };
      recognition.onspeechstart = clearSilenceTimer;
      recognition.onspeechend = scheduleRecognitionSilenceStop;
      recognition.onerror = (event) => {
        if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
          stopVoiceConversation({ discard: true });
          setError('Microphone or speech access was blocked. Allow access in your browser and try again.');
        } else if (event.error !== 'aborted' && event.error !== 'no-speech') {
          stopVoiceConversation({ discard: true });
          setError(`Live dictation stopped (${event.error}). You can try again or attach an audio file.`);
        } else {
          setListening(false);
        }
      };
      let autoSendHandled = false;
      recognition.onend = () => {
        clearSilenceTimer();
        setListening(false);
        recognitionRef.current = null;
        const transcript = [speechBaseRef.current, speechFinalRef.current]
          .map((part) => part.trim()).filter(Boolean).join(' ');
        setDraft(transcript);
        if (autoSend && !autoSendHandled) {
          autoSendHandled = true;
          if (transcript.trim()) sendMessage(transcript, { fromVoice: true });
          else if (voiceModeRef.current) scheduleVoiceListening();
          else setError('I didn’t catch a question. Try speaking again.');
        }
      };
      recognition.start();
      recognitionRef.current = recognition;
      setError('');
      setListening(true);
    } catch {
      setError('Could not start live dictation. Check microphone permission and try again.');
      setListening(false);
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
    voiceModeRef.current = true;
    setVoiceMode(true);
    setError('');
    startLiveDictation({ autoSend: true });
  }

  function stopVoiceConversation({ discard = false } = {}) {
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
      stopSilenceMonitor();
      setListening(false);
      setRecording(false);
    }
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
        if (audio.size) sendAudio(new File([audio], `recording.${extension}`, { type: audio.type }));
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

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''} ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
        <div className="sidebar-top">
          <a className="brand" href="#top" aria-label="EVA home">
            <span className="brand-mark"><AudioWaveform size={19} strokeWidth={2.2} /></span>
            <span>EVA</span>
            <span className="brand-version">PERSONAL</span>
          </a>
          <button className="icon-button sidebar-close" onClick={() => setSidebarOpen(false)} aria-label="Close menu"><X size={18} /></button>
          <button className="new-chat-button" onClick={startNewChat}>
            <MessageSquarePlus size={17} /><span>New conversation</span><span className="shortcut">Alt K</span>
          </button>
          <div className="sidebar-label">YOUR SPACE</div>
          <div className="conversation-items">
            {[...conversations].sort((first, second) => second.updatedAt - first.updatedAt).map((conversation) => (
              <button
                className={`conversation-item ${conversation.id === activeConversationId ? 'conversation-item-active' : ''}`}
                key={conversation.id}
                onClick={() => openConversation(conversation.id)}
                title={conversation.title}
              >
                <MessageSquarePlus size={15} />
                <span>{conversation.title}</span>
              </button>
            ))}
          </div>
        </div>
        <div className="sidebar-bottom">
          <button className="profile-row"><span className="avatar user-avatar">A</span><span className="profile-name">Your workspace</span><Settings2 size={16} /></button>
        </div>
      </aside>
      {sidebarOpen && <button className="scrim" onClick={() => setSidebarOpen(false)} aria-label="Close navigation" />}

      <main className="main-panel" id="top">
        <header className="topbar">
          <button className="icon-button mobile-menu" onClick={() => setSidebarOpen(true)} aria-label="Open menu"><Menu size={19} /></button>
          <button className="icon-button sidebar-toggle" onClick={() => setSidebarCollapsed((collapsed) => !collapsed)} aria-label={sidebarCollapsed ? 'Show sidebar' : 'Hide sidebar'} title={sidebarCollapsed ? 'Show sidebar' : 'Hide sidebar'}>
            {sidebarCollapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
          </button>
          <div className="model-picker"><span>EVA</span><span className="model-light">Personal assistant</span><ChevronDown size={15} /></div>
          <button className="topbar-help" title="About EVA"><CircleHelp size={18} /></button>
        </header>

        <section className={`conversation ${messages.length ? 'has-messages' : 'empty-conversation'}`}>
          {messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-mark"><AudioLines size={29} strokeWidth={1.8} /></div>
              <div className="eyebrow"><Sparkles size={13} /> A LITTLE SPACE TO THINK</div>
              <h1>What’s on your mind?</h1>
              <p>Ask anything, work through an idea, or just start talking.</p>
              <button
                className={`voice-start ${voiceMode ? 'voice-start-active' : ''}`}
                onClick={() => (voiceMode ? stopVoiceConversation() : startVoiceConversation())}
                disabled={busy && !voiceMode}
              >
                <span className="voice-start-icon">{voiceMode ? <Square size={17} fill="currentColor" /> : <Mic size={20} />}</span>
                <span className="voice-start-copy">
                  <strong>{voiceMode ? 'Stop voice conversation' : 'Start a voice conversation'}</strong>
                  <small>{busy && voiceMode ? 'EVA is replying, then it will listen again' : listening ? 'Speak naturally; pause between turns' : recording ? 'Listening for your next turn' : voiceMode ? 'Voice conversation is active' : 'Talk with EVA hands-free until you stop'}</small>
                </span>
                <span className="voice-start-hint">{voiceMode ? 'STOP' : 'VOICE'}</span>
              </button>
              <div className="starter-grid">
                {starters.map((starter, index) => (
                  <button className="starter-card" key={starter.title} onClick={() => sendMessage(starter.text)} disabled={busy}>
                    <span>{starter.title}</span><small>{starter.text}</small>
                    <span className={`starter-spark spark-${index}`}><Sparkles size={15} /></span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message) => (
                <article className={`message message-${message.role}`} key={message.id}>
                  {message.role === 'assistant' ? <span className="avatar eva-avatar"><AudioWaveform size={15} /></span> : null}
                  <div className="message-body">
                    {message.role === 'assistant' && <div className="message-author">EVA</div>}
                    <div className="message-content">{message.content}</div>
                    {message.voice && <span className="voice-label"><AudioLines size={12} /> Voice message</span>}
                  </div>
                </article>
              ))}
              {busy && <article className="message message-assistant"><span className="avatar eva-avatar"><AudioWaveform size={15} /></span><div className="typing-indicator" aria-label="EVA is thinking"><i /><i /><i /></div></article>}
              <div ref={messagesEndRef} />
            </div>
          )}
        </section>

        {voiceMode && messages.length > 0 && (
          <div className="voice-session-strip" role="status">
            <span className="voice-session-indicator"><i />{busy ? 'EVA is speaking' : listening || recording ? 'Listening for your next turn' : 'Voice conversation active'}</span>
            <button onClick={() => stopVoiceConversation()}><Square size={12} fill="currentColor" /> Stop conversation</button>
          </div>
        )}

        <div className="composer-area">
          {error && <div className="error-banner" role="alert"><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss error"><X size={15} /></button></div>}
          <form className={`composer ${recording ? 'is-recording' : ''} ${listening ? 'is-listening' : ''}`} onSubmit={(event) => { event.preventDefault(); sendMessage(); }}>
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
                <button type="button" className="composer-icon" onClick={() => fileInputRef.current?.click()} disabled={busy || recording} aria-label="Attach audio file" title="Attach audio"><Paperclip size={17} /></button>
                <button type="button" className={`composer-icon mic-button ${recording ? 'recording' : ''} ${listening ? 'listening' : ''}`} onClick={toggleMicrophone} disabled={busy} aria-label={recording || listening ? 'Stop microphone' : 'Start live dictation'} title={recording || listening ? 'Stop microphone' : 'Speak to EVA'}>
                  {recording || listening ? <Square size={14} fill="currentColor" /> : <Mic size={17} />}
                </button>
              </div>
              <button type="submit" className="send-button" disabled={busy || !draft.trim() || recording || listening} aria-label="Send message"><Send size={16} /></button>
            </div>
          </form>
          <div className="composer-footnote">{listening ? <span className="live-status"><i /> Listening live · click stop when finished</span> : <span><Check size={12} /> Chat history saved in this browser</span>}<span>Enter to send · Shift + Enter for a new line</span></div>
        </div>
      </main>
      <input ref={fileInputRef} className="visually-hidden" type="file" accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac,.webm" onChange={(event) => sendAudio(event.target.files?.[0])} />
    </div>
  );
}
