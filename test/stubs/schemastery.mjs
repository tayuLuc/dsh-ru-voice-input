/**
 * Stand-in for @deepseek-ai/schemastery.
 *
 * The registry's Config schema is a declaration the host applies, not logic
 * the tests exercise, so every builder here returns a chainable descriptor.
 * `string()` returns an object with the members the real schema uses: min,
 * required, default, and volatile.
 */
const descriptor = () => {
	const self = {
		min: () => self,
		required: () => self,
		default: () => self,
		volatile: () => self
	};
	return self;
};

export default {
	object: () => descriptor(),
	string: () => descriptor(),
	natural: () => descriptor(),
	boolean: () => descriptor(),
	union: () => descriptor()
};
