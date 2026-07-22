" SPDX-License-Identifier: MIT
" Copyright (c) 2026 Jarkko Sakkinen

vim9script

export const PluginVersion = '0.8.4'
const HealthCacheKey = 'readseek_health'

export def Initialize()
  SetDefault('readseek_root_markers', ['.git'])
  SetDefault('readseek_list_type', 'quickfix')
  SetDefault('readseek_auto_install', false)
  SetDefault('readseek_auto_open_results', true)
  SetDefault('readseek_job_timeout_ms', 30000)
  SetDefault('readseek_notification_timeout_ms', 4000)
  Validate()
enddef

def SetDefault(name: string, value: any)
  if !exists($'g:{name}')
    g:[name] = value
  endif
enddef

export def Validate()
  if type(get(g:, 'readseek_root_markers', [])) != v:t_list
    throw 'readseek.vim: g:readseek_root_markers must be a list'
  endif
  if index(['quickfix', 'location'], get(g:, 'readseek_list_type', '')) < 0
    throw 'readseek.vim: g:readseek_list_type must be quickfix or location'
  endif
  if type(get(g:, 'readseek_auto_install', false)) != v:t_bool
    throw 'readseek.vim: g:readseek_auto_install must be a boolean'
  endif
  if type(get(g:, 'readseek_auto_open_results', true)) != v:t_bool
    throw 'readseek.vim: g:readseek_auto_open_results must be a boolean'
  endif
  if type(get(g:, 'readseek_job_timeout_ms', 0)) != v:t_number || g:readseek_job_timeout_ms <= 0
    throw 'readseek.vim: g:readseek_job_timeout_ms must be a positive number'
  endif
  if type(get(g:, 'readseek_notification_timeout_ms', 0)) != v:t_number || g:readseek_notification_timeout_ms <= 0
    throw 'readseek.vim: g:readseek_notification_timeout_ms must be a positive number'
  endif
enddef

export def LocalBinaryPath(): string
  if has('win32')
    return expand('$APPDATA') .. '\readseek.vim\bin\readseek.exe'
  endif
  return expand('~/.local/share/readseek.vim/bin/readseek')
enddef

export def LocalBinaryDir(): string
  return fnamemodify(LocalBinaryPath(), ':h')
enddef

export def ExecutablePath(): string
  var executable = get(g:, 'readseek_executable', '')
  if !empty(executable)
    return executable
  endif
  return LocalBinaryPath()
enddef

export def IsExecutableAvailable(): bool
  return executable(ExecutablePath()) == 1
enddef

export def JobTimeout(): number
  return g:readseek_job_timeout_ms
enddef

export def AutoOpenResults(): bool
  return g:readseek_auto_open_results
enddef

export def NotificationTimeout(): number
  return g:readseek_notification_timeout_ms
enddef

export def AutoInstall(): bool
  return g:readseek_auto_install
enddef

export def InvalidateHealthCache()
  unlet! g:[HealthCacheKey]
enddef

export def VersionAt(path: string): string
  var output = systemlist(shellescape(path) .. ' --version')
  if v:shell_error != 0 || empty(output)
    return ''
  endif
  return matchstr(output[0], '\\v\\d+\\.\\d+\\.\\d+')
enddef

export def Version(): string
  return VersionAt(ExecutablePath())
enddef

export def IsHealthCached(): bool
  var cache = get(g:, HealthCacheKey, v:null)
  return type(cache) == v:t_dict && get(cache, 'path', '') ==# ExecutablePath()
enddef

export def CacheHealth(version: string)
  g:[HealthCacheKey] = {path: ExecutablePath(), version: version}
enddef

export def CheckHealth(): dict<any>
  if IsHealthCached()
    return {ok: true, message: 'readseek.vim: readseek health check already passed'}
  endif
  if !IsExecutableAvailable()
    return {ok: false, message: 'readseek.vim: binary not installed'}
  endif

  var version = Version()
  var path = ExecutablePath()
  if version !=# PluginVersion
    return {ok: false, message: $'readseek.vim: expected readseek {PluginVersion}, found {version} at {path}'}
  endif
  CacheHealth(version)
  return {ok: true, message: $'readseek.vim: readseek {version} found at {path}'}
enddef