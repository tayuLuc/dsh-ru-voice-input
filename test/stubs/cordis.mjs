/**
 * Stand-in for @deepseek-ai/cordis: the registry only needs a base class that
 * records the service name and keeps the context.
 */
export class Service {
	constructor(ctx, name) {
		this.ctx = ctx;
		this.serviceName = name;
	}
}
