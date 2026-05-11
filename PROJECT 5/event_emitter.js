/**
 * EventEmitter — a lightweight, dependency-free pub/sub event bus.
 *
 * Supports multiple listeners per event, one-time listeners,
 * listener removal, and wildcard '*' events.
 *
 * @module EventEmitter
 * @example
 * const emitter = new EventEmitter();
 * emitter.on('data', (payload) => console.log(payload));
 * emitter.emit('data', { value: 42 });
 */
class EventEmitter {
  constructor() {
    /** @private @type {Object.<string, Function[]>} */
    this._events = {};
    /** @private @type {number} */
    this._maxListeners = 10;
  }

  /**
   * Set the maximum number of listeners per event (default: 10).
   * Emits a warning if exceeded.
   * @param {number} n - Maximum listener count.
   * @returns {EventEmitter} this (chainable)
   */
  setMaxListeners(n) {
    this._maxListeners = n;
    return this;
  }

  /**
   * Register a listener for a named event.
   * @param {string} event - Event name (use '*' for wildcard).
   * @param {Function} listener - Callback invoked when the event fires.
   * @returns {EventEmitter} this (chainable)
   * @throws {TypeError} If listener is not a function.
   */
  on(event, listener) {
    if (typeof listener !== 'function') throw new TypeError('Listener must be a function');
    if (!this._events[event]) this._events[event] = [];
    if (this._events[event].length >= this._maxListeners) {
      console.warn(`MaxListenersExceededWarning: ${event} has more than ${this._maxListeners} listeners`);
    }
    this._events[event].push(listener);
    return this;
  }

  /**
   * Register a one-time listener that auto-removes after first invocation.
   * @param {string} event - Event name.
   * @param {Function} listener - One-time callback.
   * @returns {EventEmitter} this
   */
  once(event, listener) {
    const wrapper = (...args) => { listener(...args); this.off(event, wrapper); };
    wrapper._original = listener;
    return this.on(event, wrapper);
  }

  /**
   * Remove a specific listener from an event.
   * @param {string} event - Event name.
   * @param {Function} listener - The listener to remove.
   * @returns {EventEmitter} this
   */
  off(event, listener) {
    if (!this._events[event]) return this;
    this._events[event] = this._events[event].filter(
      l => l !== listener && l._original !== listener
    );
    return this;
  }

  /**
   * Remove ALL listeners for a given event, or all events if omitted.
   * @param {string} [event] - Event name. If omitted, clears everything.
   * @returns {EventEmitter} this
   */
  removeAllListeners(event) {
    if (event) delete this._events[event];
    else this._events = {};
    return this;
  }

  /**
   * Emit an event, synchronously calling all registered listeners.
   * Also triggers wildcard '*' listeners (receives event name + args).
   * @param {string} event - Event name.
   * @param {...*} args - Arguments forwarded to each listener.
   * @returns {boolean} True if at least one listener was called.
   */
  emit(event, ...args) {
    let called = false;
    if (this._events[event]?.length) {
      [...this._events[event]].forEach(l => l(...args));
      called = true;
    }
    if (event !== '*' && this._events['*']?.length) {
      [...this._events['*']].forEach(l => l(event, ...args));
      called = true;
    }
    return called;
  }

  /**
   * Return the list of listeners registered for an event.
   * @param {string} event
   * @returns {Function[]}
   */
  listeners(event) {
    return [...(this._events[event] || [])];
  }

  /**
   * Return the number of listeners for an event.
   * @param {string} event
   * @returns {number}
   */
  listenerCount(event) {
    return (this._events[event] || []).length;
  }
}

module.exports = EventEmitter;
