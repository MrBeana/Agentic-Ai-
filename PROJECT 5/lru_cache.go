// Package cache provides a thread-safe, in-memory LRU cache with optional TTL support.
//
// It is designed for low-latency caching of frequently accessed data in
// concurrent Go services. Entries can expire automatically based on a
// configured time-to-live (TTL), or persist indefinitely (TTL = 0).
//
// Usage:
//
//	c := cache.New(1000, 5*time.Minute)
//	c.Set("key", myValue)
//	val, ok := c.Get("key")
package cache

import (
	"container/list"
	"sync"
	"time"
)

// entry is an internal cache record stored in the LRU list.
type entry struct {
	key       string
	value     interface{}
	expiresAt time.Time
}

// Cache is a thread-safe LRU cache with optional TTL-based expiration.
type Cache struct {
	mu       sync.Mutex
	items    map[string]*list.Element
	lru      *list.List
	maxItems int
	ttl      time.Duration
	hits     int64
	misses   int64
}

// New creates and returns a new Cache.
//
// Parameters:
//   - maxItems: maximum number of items to hold before evicting the LRU entry (must be > 0).
//   - ttl: time-to-live for each entry; use 0 for no expiration.
//
// Returns a pointer to the initialized Cache.
func New(maxItems int, ttl time.Duration) *Cache {
	if maxItems <= 0 {
		maxItems = 128
	}
	return &Cache{
		items:    make(map[string]*list.Element),
		lru:      list.New(),
		maxItems: maxItems,
		ttl:      ttl,
	}
}

// Set stores a key-value pair in the cache.
// If the key already exists its value is updated and it is moved to the front.
// If the cache is at capacity the least-recently-used item is evicted.
func (c *Cache) Set(key string, value interface{}) {
	c.mu.Lock()
	defer c.mu.Unlock()

	exp := time.Time{}
	if c.ttl > 0 {
		exp = time.Now().Add(c.ttl)
	}

	if el, ok := c.items[key]; ok {
		c.lru.MoveToFront(el)
		el.Value.(*entry).value = value
		el.Value.(*entry).expiresAt = exp
		return
	}

	if c.lru.Len() >= c.maxItems {
		c.evict()
	}

	e := &entry{key: key, value: value, expiresAt: exp}
	el := c.lru.PushFront(e)
	c.items[key] = el
}

// Get retrieves a value from the cache by key.
// The entry is moved to the front of the LRU list on access.
//
// Returns (value, true) if found and not expired; (nil, false) otherwise.
func (c *Cache) Get(key string) (interface{}, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	el, ok := c.items[key]
	if !ok {
		c.misses++
		return nil, false
	}

	e := el.Value.(*entry)
	if !e.expiresAt.IsZero() && time.Now().After(e.expiresAt) {
		c.removeElement(el)
		c.misses++
		return nil, false
	}

	c.lru.MoveToFront(el)
	c.hits++
	return e.value, true
}

// Delete removes a key from the cache if it exists.
func (c *Cache) Delete(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if el, ok := c.items[key]; ok {
		c.removeElement(el)
	}
}

// Flush removes all entries from the cache.
func (c *Cache) Flush() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items = make(map[string]*list.Element)
	c.lru.Init()
}

// Len returns the current number of items in the cache (including expired ones not yet evicted).
func (c *Cache) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.lru.Len()
}

// Stats returns a snapshot of cache hit/miss counts.
func (c *Cache) Stats() (hits, misses int64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.hits, c.misses
}

// evict removes the least-recently-used item. Must be called with c.mu held.
func (c *Cache) evict() {
	el := c.lru.Back()
	if el != nil {
		c.removeElement(el)
	}
}

// removeElement removes a list element and its map entry. Must be called with c.mu held.
func (c *Cache) removeElement(el *list.Element) {
	c.lru.Remove(el)
	e := el.Value.(*entry)
	delete(c.items, e.key)
}
