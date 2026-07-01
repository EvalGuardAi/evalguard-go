# Releasing the Go SDK

The Go SDK is published from the public mirror repo
`github.com/EvalGuardAi/evalguard-go` (the EvalGuard monorepo where this
source lives is private, so `go get` cannot resolve it directly). Tags
on the public mirror are picked up by `proxy.golang.org` automatically
— no centralized registry, no auth needed.

## Live status

As of 2026-04-28, `proxy.golang.org` lists `v1.0.0`, `v1.0.1`, `v1.0.2`,
`v1.0.3`, `v1.0.4`. `pkg.go.dev/github.com/EvalGuardAi/evalguard-go`
returns 200. Consumers use:

```bash
go get github.com/EvalGuardAi/evalguard-go@latest
```

## Mirroring from the monorepo

Since the monorepo is private, every release needs `packages/go-sdk` to
be pushed onto the public `evalguard-go` repo before the tag is cut.
The standard one-shot:

```bash
# Inside the monorepo root, with `evalguard-go` as a remote:
git subtree push --prefix=packages/go-sdk evalguard-go main
```

For a hands-off setup, add a `release/sync-go-mirror.yml` workflow that
runs `git subtree push` on every push to `main` so the public mirror
never lags. Until that exists, the operator must run the subtree push
manually before tagging.

## Cutting a release

1. Push the subtree to `evalguard-go` (see above).
2. Inside the `evalguard-go` repo, ensure `go test ./...` is green.
3. Tag and push:
   ```bash
   cd /path/to/evalguard-go
   git tag v1.2.3
   git push origin --tags
   ```
4. `proxy.golang.org` picks up the new version within ~1 minute. The
   `publish-go-sdk.yml` workflow in the monorepo can also be triggered
   on a `go-sdk-v1.2.3` tag for proxy-warming + release notes, but the
   primary publish event is the tag on the mirror repo itself.
5. Verify:
   ```bash
   go get github.com/EvalGuardAi/evalguard-go@v1.2.3
   ```

## Major versions (v2+)

If bumping to `v2.x.x`, the module path in `go.mod` must be suffixed with `/v2`
and consumers must import the new path. See https://go.dev/doc/modules/major-version.
