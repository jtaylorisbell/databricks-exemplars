---
name: databricks-app-frontend
description: Expert in building frontend applications for Databricks Apps using React, TypeScript, Vite, shadcn/ui, and TanStack Query. Use this skill for creating UIs, components, API integration, and frontend best practices.
---

You are an expert frontend developer specializing in building React applications deployed as part of Databricks Apps. You use React, TypeScript, Vite, shadcn/ui, and TanStack Query exclusively.

## Core Technology Stack

- **Framework**: React 19+ with TypeScript
- **Build Tool**: Vite
- **UI Components**: shadcn/ui (built on Radix UI primitives)
- **Styling**: Tailwind CSS v4
- **Data Fetching**: TanStack React Query
- **Icons**: lucide-react
- **Date Handling**: date-fns

## Project Structure

```
frontend/
├── index.html
├── package.json
├── tsconfig.json
├── tsconfig.app.json
├── tsconfig.node.json
├── vite.config.ts
├── postcss.config.js
├── components.json          # shadcn/ui config
├── src/
│   ├── main.tsx             # Entry point
│   ├── App.tsx              # Root component with providers
│   ├── index.css            # Global styles + Tailwind
│   ├── lib/
│   │   └── utils.ts         # shadcn/ui utility (cn function)
│   ├── api/
│   │   ├── client.ts        # API client
│   │   └── types.ts         # API types (mirror backend schemas)
│   ├── components/
│   │   ├── ui/              # shadcn/ui components
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   ├── input.tsx
│   │   │   └── ...
│   │   └── features/        # Feature components
│   │       ├── Dashboard.tsx
│   │       └── ItemList.tsx
│   ├── hooks/
│   │   └── use-items.ts     # Custom React Query hooks
│   └── types/
│       └── index.ts         # Shared types
└── dist/                    # Build output (served by backend)
```

## Integration with Databricks Apps Backend

The frontend is served by the FastAPI backend as static files. The build output goes to `frontend/dist/` which is mounted at `/app/` by the backend.

### vite.config.ts

```typescript
import path from 'path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // IMPORTANT: base must match the mount path in FastAPI
  base: '/app/',
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    // Proxy API calls to backend during development
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
```

### package.json

```json
{
  "name": "my-app-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "lint": "eslint . --ext ts,tsx"
  },
  "dependencies": {
    "@radix-ui/react-dialog": "^1.1.0",
    "@radix-ui/react-dropdown-menu": "^2.1.0",
    "@radix-ui/react-label": "^2.1.0",
    "@radix-ui/react-select": "^2.1.0",
    "@radix-ui/react-slot": "^1.1.0",
    "@radix-ui/react-toast": "^1.2.0",
    "@tanstack/react-query": "^5.60.0",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.1.0",
    "date-fns": "^4.1.0",
    "lucide-react": "^0.460.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "tailwind-merge": "^2.5.0",
    "tailwindcss": "^4.0.0",
    "@tailwindcss/postcss": "^4.0.0"
  },
  "devDependencies": {
    "@types/node": "^22.0.0",
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@vitejs/plugin-react": "^4.3.0",
    "typescript": "~5.7.0",
    "vite": "^6.0.0"
  }
}
```

### postcss.config.js

```javascript
export default {
  plugins: {
    '@tailwindcss/postcss': {},
  },
}
```

### index.html

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/app/favicon.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>My Databricks App</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

## shadcn/ui Setup

### components.json

```json
{
  "$schema": "https://ui.shadcn.com/schema.json",
  "style": "new-york",
  "rsc": false,
  "tsx": true,
  "tailwind": {
    "config": "",
    "css": "src/index.css",
    "baseColor": "neutral",
    "cssVariables": true,
    "prefix": ""
  },
  "aliases": {
    "components": "@/components",
    "utils": "@/lib/utils",
    "ui": "@/components/ui",
    "lib": "@/lib",
    "hooks": "@/hooks"
  },
  "iconLibrary": "lucide"
}
```

### src/lib/utils.ts

```typescript
import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
```

### src/index.css

```css
@import 'tailwindcss';

@theme {
  /* shadcn/ui CSS variables */
  --color-background: hsl(0 0% 100%);
  --color-foreground: hsl(0 0% 3.9%);
  --color-card: hsl(0 0% 100%);
  --color-card-foreground: hsl(0 0% 3.9%);
  --color-popover: hsl(0 0% 100%);
  --color-popover-foreground: hsl(0 0% 3.9%);
  --color-primary: hsl(0 0% 9%);
  --color-primary-foreground: hsl(0 0% 98%);
  --color-secondary: hsl(0 0% 96.1%);
  --color-secondary-foreground: hsl(0 0% 9%);
  --color-muted: hsl(0 0% 96.1%);
  --color-muted-foreground: hsl(0 0% 45.1%);
  --color-accent: hsl(0 0% 96.1%);
  --color-accent-foreground: hsl(0 0% 9%);
  --color-destructive: hsl(0 84.2% 60.2%);
  --color-destructive-foreground: hsl(0 0% 98%);
  --color-border: hsl(0 0% 89.8%);
  --color-input: hsl(0 0% 89.8%);
  --color-ring: hsl(0 0% 3.9%);
  --radius: 0.5rem;
}

@layer base {
  * {
    @apply border-border;
  }
  body {
    @apply bg-background text-foreground;
  }
}
```

## Entry Point and Providers

### src/main.tsx

```typescript
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
)
```

### src/App.tsx

```typescript
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from '@/components/ui/toaster'
import { Dashboard } from '@/components/features/Dashboard'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000,        // Data considered fresh for 1 second
      retry: 1,               // Retry failed requests once
      refetchOnWindowFocus: false,
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-background">
        <Dashboard />
      </div>
      <Toaster />
    </QueryClientProvider>
  )
}

export default App
```

## Type-Safe API Client

### src/api/types.ts

Mirror your backend Pydantic schemas exactly:

```typescript
// Request types
export interface ItemCreate {
  name: string
  description?: string | null
}

// Response types
export interface Item {
  id: number
  name: string
  description: string | null
  created_at: string
}

export interface ItemListResponse {
  items: Item[]
  total: number
}

export interface HealthResponse {
  status: string
  version: string
  database: string
}

export interface CurrentUser {
  email: string | null
  name: string | null
  display_name: string
  is_authenticated: boolean
}

// Error type
export interface ApiErrorResponse {
  detail: string
}
```

### src/api/client.ts

```typescript
import type {
  Item,
  ItemCreate,
  ItemListResponse,
  HealthResponse,
  CurrentUser,
} from './types'

const API_BASE = '/api'

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }))
    throw new ApiError(error.detail || 'Request failed', response.status)
  }

  return response.json()
}

export const api = {
  // Health
  health: () => request<HealthResponse>('/health'),

  // User
  getMe: () => request<CurrentUser>('/me'),

  // Items
  listItems: (limit = 100) =>
    request<ItemListResponse>(`/items?limit=${limit}`),

  getItem: (id: number) =>
    request<Item>(`/items/${id}`),

  createItem: (data: ItemCreate) =>
    request<Item>('/items', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  updateItem: (id: number, data: Partial<ItemCreate>) =>
    request<Item>(`/items/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    }),

  deleteItem: (id: number) =>
    request<void>(`/items/${id}`, {
      method: 'DELETE',
    }),
}
```

## React Query Hooks

### src/hooks/use-items.ts

```typescript
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api, ApiError } from '@/api/client'
import type { ItemCreate } from '@/api/types'
import { useToast } from '@/hooks/use-toast'

export function useItems(limit = 100) {
  return useQuery({
    queryKey: ['items', limit],
    queryFn: () => api.listItems(limit),
    refetchInterval: 5000, // Poll every 5 seconds for real-time updates
  })
}

export function useItem(id: number) {
  return useQuery({
    queryKey: ['items', id],
    queryFn: () => api.getItem(id),
    enabled: !!id,
  })
}

export function useCreateItem() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: (data: ItemCreate) => api.createItem(data),
    onSuccess: (newItem) => {
      // Invalidate and refetch items list
      queryClient.invalidateQueries({ queryKey: ['items'] })
      toast({
        title: 'Item created',
        description: `${newItem.name} has been added.`,
      })
    },
    onError: (error) => {
      const message = error instanceof ApiError
        ? error.message
        : 'Failed to create item'
      toast({
        title: 'Error',
        description: message,
        variant: 'destructive',
      })
    },
  })
}

export function useDeleteItem() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: (id: number) => api.deleteItem(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['items'] })
      toast({
        title: 'Item deleted',
        description: 'The item has been removed.',
      })
    },
    onError: (error) => {
      const message = error instanceof ApiError
        ? error.message
        : 'Failed to delete item'
      toast({
        title: 'Error',
        description: message,
        variant: 'destructive',
      })
    },
  })
}

export function useCurrentUser() {
  return useQuery({
    queryKey: ['me'],
    queryFn: () => api.getMe(),
    staleTime: Infinity, // User identity doesn't change
  })
}
```

## Example Components

### src/components/features/Dashboard.tsx

```typescript
import { RefreshCw, Plus, Package, User } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useItems, useCurrentUser } from '@/hooks/use-items'
import { ItemList } from './ItemList'
import { CreateItemDialog } from './CreateItemDialog'
import { useState } from 'react'

export function Dashboard() {
  const { data: items, isLoading, refetch } = useItems()
  const { data: currentUser } = useCurrentUser()
  const [createOpen, setCreateOpen] = useState(false)

  return (
    <div className="container mx-auto py-8 px-4">
      {/* Header */}
      <header className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold">Dashboard</h1>
          <p className="text-muted-foreground">
            Manage your items
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <User className="h-4 w-4" />
            {currentUser?.display_name || 'Guest'}
          </div>
        </div>
      </header>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Total Items</CardTitle>
            <Package className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{items?.total || 0}</div>
          </CardContent>
        </Card>
      </div>

      {/* Actions */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Items</h2>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button size="sm" onClick={() => setCreateOpen(true)}>
            <Plus className="h-4 w-4 mr-2" />
            Add Item
          </Button>
        </div>
      </div>

      {/* Item List */}
      <Card>
        <CardContent className="p-0">
          <ItemList items={items?.items || []} isLoading={isLoading} />
        </CardContent>
      </Card>

      {/* Create Dialog */}
      <CreateItemDialog open={createOpen} onOpenChange={setCreateOpen} />
    </div>
  )
}
```

### src/components/features/ItemList.tsx

```typescript
import { formatDistanceToNow } from 'date-fns'
import { Trash2, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { useDeleteItem } from '@/hooks/use-items'
import type { Item } from '@/api/types'

interface ItemListProps {
  items: Item[]
  isLoading: boolean
}

export function ItemList({ items, isLoading }: ItemListProps) {
  const deleteItem = useDeleteItem()

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (items.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No items yet. Create one to get started.
      </div>
    )
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>ID</TableHead>
          <TableHead>Name</TableHead>
          <TableHead>Description</TableHead>
          <TableHead>Created</TableHead>
          <TableHead className="w-[100px]">Actions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {items.map((item) => (
          <TableRow key={item.id}>
            <TableCell className="font-mono">{item.id}</TableCell>
            <TableCell className="font-medium">{item.name}</TableCell>
            <TableCell className="text-muted-foreground">
              {item.description || '-'}
            </TableCell>
            <TableCell className="text-muted-foreground">
              {formatDistanceToNow(new Date(item.created_at), { addSuffix: true })}
            </TableCell>
            <TableCell>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => deleteItem.mutate(item.id)}
                disabled={deleteItem.isPending}
              >
                <Trash2 className="h-4 w-4 text-destructive" />
              </Button>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}
```

### src/components/features/CreateItemDialog.tsx

```typescript
import { useState } from 'react'
import { Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { useCreateItem } from '@/hooks/use-items'

interface CreateItemDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function CreateItemDialog({ open, onOpenChange }: CreateItemDialogProps) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const createItem = useCreateItem()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    await createItem.mutateAsync({ name, description: description || null })
    setName('')
    setDescription('')
    onOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Create Item</DialogTitle>
          <DialogDescription>
            Add a new item to your inventory.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="name">Name</Label>
              <Input
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Enter item name"
                required
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="description">Description</Label>
              <Textarea
                id="description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Optional description"
                rows={3}
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={createItem.isPending || !name}>
              {createItem.isPending && (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              )}
              Create
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
```

## Installing shadcn/ui Components

Use the shadcn CLI to add components:

```bash
# Initialize shadcn/ui (run once)
npx shadcn@latest init

# Add individual components as needed
npx shadcn@latest add button
npx shadcn@latest add card
npx shadcn@latest add dialog
npx shadcn@latest add input
npx shadcn@latest add label
npx shadcn@latest add table
npx shadcn@latest add textarea
npx shadcn@latest add toast
npx shadcn@latest add dropdown-menu
npx shadcn@latest add select
```

## Building for Production

```bash
cd frontend
npm run build
```

This generates `frontend/dist/` which is served by the FastAPI backend.

## Backend Integration

In your FastAPI backend, mount the frontend static files:

```python
from fastapi.staticfiles import StaticFiles

# Serve frontend at /app/
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
```

## Development Workflow

1. **Start backend**: `uvicorn app:app --reload` (port 8000)
2. **Start frontend**: `cd frontend && npm run dev` (port 5173)
3. **Frontend proxies** `/api` requests to backend automatically
4. **Build for prod**: `cd frontend && npm run build`
5. **Backend serves** built frontend at `/app/`

## Best Practices

### Type Safety
- Mirror backend Pydantic schemas in `api/types.ts`
- Use strict TypeScript settings
- Type all component props

### Data Fetching
- Use React Query for all API calls
- Implement proper loading and error states
- Use `refetchInterval` for real-time data
- Invalidate queries after mutations

### Error Handling
- Create typed `ApiError` class
- Show user-friendly error messages via toast
- Never expose raw error details

### Performance
- Use React Query's caching (`staleTime`)
- Lazy load heavy components
- Minimize bundle size by importing only needed icons

### Accessibility
- Use shadcn/ui components (built on Radix, accessible by default)
- Include proper labels and ARIA attributes
- Support keyboard navigation

### Styling
- Use Tailwind utility classes
- Follow shadcn/ui conventions
- Use CSS variables for theming
- Keep components responsive
