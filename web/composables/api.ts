import { useCookie, useRuntimeConfig } from "nuxt/app";
import {
  createPromiseClient,
  type Interceptor,
  type Transport,
} from "@connectrpc/connect";
import { createGrpcWebTransport } from "@connectrpc/connect-web";
import type { GrpcTransportOptions } from "@connectrpc/connect-node";
import { Chronotopia } from "~/model/chronotopia_connect";
import { UserManagement } from "~/model/user_management_connect";
import { Ingest } from "~/model/ingest_connect";
import { Trips } from "~/model/trips_connect";

// WATCH OUT
// Even though recent browsers should support it, loading the web GRPC transport at runtime
// only works in dev and not in prod (might be Nuxt/Vite/... bug). By loading and later
// overwriting the transport constructor serverside we take a minor hit but at least it works.
// Moral of the story: do not `await import()` clientside
let createTransport: (options: GrpcTransportOptions) => Transport =
  createGrpcWebTransport;

if (!import.meta.client) {
  createTransport = (await import("@connectrpc/connect-node"))
    .createGrpcTransport;
}

let transport: Transport;

export const useApi = () => {
  const cfg = useRuntimeConfig();
  const baseUrl = import.meta.client ? cfg.public.apiBaseUrl : cfg.apiBaseUrl;

  const makeInterceptor = (): Interceptor => {
    const token = useCookie<string>("auth-token");
    return (next) => async (req) => {
      if (token.value !== undefined) {
        req.header.append("authorization", token.value);
      } else {
        req.header.append(
          "authorization",
          "Bearer 1a7727de-8c30-4af9-befc-10681b8d5493"
        );
      }

      const res = await next(req);
      return res;
    };
  };

  if (!import.meta.client) {
    transport = createTransport({
      baseUrl,
      httpVersion: "2",
      interceptors: [makeInterceptor()],
    });
  } else {
    // @ts-expect-error it is the node ctor here without http
    transport = createTransport({
      baseUrl,
      interceptors: [makeInterceptor()],
    });
  }

  const api = createPromiseClient(Chronotopia, transport);
  const userManagementApi = createPromiseClient(UserManagement, transport);
  const ingestApi = createPromiseClient(Ingest, transport);
  const tripsApi = createPromiseClient(Trips, transport);

  return {
    api,
    userManagementApi,
    ingestApi,
    tripsApi,
  };
};
